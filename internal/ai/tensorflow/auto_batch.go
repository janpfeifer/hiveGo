package tensorflow

// This file implements the auto-batching strategy for
// tensorflow models.

import (
	"github.com/janpfeifer/hiveGo/internal/features"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"log"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type AutoBatchRequest struct {
	// Boards
	boardFeatures     []float32
	fullBoardFeatures [][][]float32 // [batch, height, width, depth]

	// Actions
	actionsIsMove       []bool
	actionsSrcPositions [][2]int64
	actionsTgtPositions [][2]int64
	actionsPieces       [][NumPieceTypes]float32

	// Channel is closed when done.
	done chan bool

	// Results
	score        float32
	actionsProbs []float32
}

func (s *Scorer) newAutoBatchRequest(b *Board, scoreActions bool) (req *AutoBatchRequest) {
	req = &AutoBatchRequest{
		boardFeatures: features.FeatureVector(b, s.version),
		done:          make(chan bool),
	}
	if s.HasFullBoard() {
		req.fullBoardFeatures = features.MakeFullBoardFeatures(b, features.SuggestedFullBoardWidth, features.SuggestedFullBoardHeight)
	}
	scoreActions = scoreActions && s.IsActionsClassifier()
	if scoreActions && b.NumActions() > 1 {
		req.actionsIsMove = make([]bool, b.NumActions())
		req.actionsSrcPositions = make([][2]int64, b.NumActions())
		req.actionsTgtPositions = make([][2]int64, b.NumActions())
		req.actionsPieces = make([][NumPieceTypes]float32, b.NumActions())
		for actionIdx, action := range b.Derived.Actions {
			req.actionsIsMove[actionIdx] = action.Move
			req.actionsSrcPositions[actionIdx] = features.PosToFullBoardPosition(b, action.SourcePos)
			req.actionsTgtPositions[actionIdx] = features.PosToFullBoardPosition(b, action.TargetPos)
			req.actionsPieces[actionIdx][int(action.Piece)-1] = 1
		}
	}
	return
}

func (req *AutoBatchRequest) LenActions() int { return len(req.actionsIsMove) }

func (s *Scorer) scoreAutoBatch(b *Board, scoreActions bool) (score float32, actionsProbs []float32) {
	// Send request and wait for it to be processed.
	req := s.newAutoBatchRequest(b, scoreActions)
	klog.V(3).Info("Sending request", s)
	s.autoBatchChan <- req
	<-req.done
	return req.score, req.actionsProbs
}

// Special request that indicates update on batch size.
var onBatchSizeUpdate = &AutoBatchRequest{}

func (s *Scorer) SetBatchSize(batchSize int) {
	if batchSize < 1 {
		batchSize = 1
	}
	s.autoBatchSize = batchSize
	s.autoBatchChan <- onBatchSizeUpdate
}

type AutoBatch struct {
	requests []*AutoBatchRequest

	// Boards
	boardFeatures     [][]float32
	fullBoardFeatures [][][][]float32 // [batch, height, width, depth]

	// Actions
	actionsBoardIndices []int64 // Go tensorflow implementation is broken for int32.
	actionsIsMove       []bool
	actionsSrcPositions [][2]int64
	actionsTgtPositions [][2]int64
	actionsPieces       [][NumPieceTypes]float32
}

const MAX_ACTIONS_PER_BOARD = 200

func (s *Scorer) newAutoBatch() *AutoBatch {
	maxActions := s.autoBatchSize * MAX_ACTIONS_PER_BOARD
	ab := &AutoBatch{
		boardFeatures: make([][]float32, 0, s.autoBatchSize),
	}
	if s.HasFullBoard() {
		ab.fullBoardFeatures = make([][][][]float32, 0, s.autoBatchSize)
	}
	if s.IsActionsClassifier() {
		ab.actionsBoardIndices = make([]int64, 0, maxActions)
		ab.actionsIsMove = make([]bool, 0, maxActions)
		ab.actionsSrcPositions = make([][2]int64, 0, maxActions)
		ab.actionsTgtPositions = make([][2]int64, 0, maxActions)
		ab.actionsPieces = make([][NumPieceTypes]float32, 0, maxActions)
	}
	return ab
}

func (ab *AutoBatch) Append(req *AutoBatchRequest) {
	requestIdx := int64(ab.Len())
	ab.requests = append(ab.requests, req)
	ab.boardFeatures = append(ab.boardFeatures, req.boardFeatures)
	if ab.fullBoardFeatures != nil {
		ab.fullBoardFeatures = append(ab.fullBoardFeatures, req.fullBoardFeatures)
	}
	if ab.actionsBoardIndices != nil && req.actionsIsMove != nil {
		for _ = range req.actionsIsMove {
			ab.actionsBoardIndices = append(ab.actionsBoardIndices, requestIdx)
		}
		ab.actionsIsMove = append(ab.actionsIsMove, req.actionsIsMove...)
		ab.actionsSrcPositions = append(ab.actionsSrcPositions, req.actionsSrcPositions...)
		ab.actionsTgtPositions = append(ab.actionsTgtPositions, req.actionsTgtPositions...)
		ab.actionsPieces = append(ab.actionsPieces, req.actionsPieces...)
	}
}

func (ab *AutoBatch) Len() int { return len(ab.requests) }

func (ab *AutoBatch) LenActions() int { return len(ab.actionsBoardIndices) }

func (s *Scorer) autoBatchScoreAndDeliver(ab *AutoBatch) {
	// Convert Go slices to tensors.
	feeds := map[tf.Output]*tf.Tensor{
		s.BoardFeatures: mustTensor(ab.boardFeatures),
	}
	if s.HasFullBoard() {
		feeds[s.FullBoard] = mustTensor(ab.fullBoardFeatures)
	}
	if s.IsActionsClassifier() && ab.LenActions() > 0 {
		feeds[s.ActionsBoardIndices] = mustTensor(ab.actionsBoardIndices)
		feeds[s.ActionsIsMove] = mustTensor(ab.actionsIsMove)
		feeds[s.ActionsSrcPositions] = mustTensor(ab.actionsSrcPositions)
		feeds[s.ActionsTgtPositions] = mustTensor(ab.actionsTgtPositions)
		feeds[s.ActionsPieces] = mustTensor(ab.actionsPieces)
	}
	if s.IsTraining.Op != nil {
		feeds[s.IsTraining] = mustTensor(false)
	}
	for _, pair := range s.Params {
		feeds[pair.key] = pair.value
	}
	fetches := []tf.Output{s.BoardPredictions}
	if s.IsActionsClassifier() && ab.LenActions() > 0 {
		fetches = append(fetches, s.ActionsPredictions)
	}
	// Evaluate: at most one evaluation at a same time.
	if klog.V(3) {
		klog.V(3).Infof("Tensors fed: ")
		for to, tensor := range feeds {
			klog.V(3).Infof("\t%s: %v", to.Op.Name(), tensor.Shape())
		}
	}

	sess := s.NextSession()
	results, err := sess.Run(feeds, fetches, nil)
	if err != nil {
		log.Panicf("Prediction failed: %v", err)
	}

	// Clone over resulting scores.
	scores := results[0].Value().([]float32)
	if len(scores) != ab.Len() {
		log.Panicf("Expected %d scores (=number of boards given), got %d",
			ab.Len(), len(scores))
	}
	for ii, score := range scores {
		ab.requests[ii].score = score
	}

	// Clone over resulting action probabilities
	if s.IsActionsClassifier() && ab.LenActions() > 0 {
		allActionsProbs := results[1].Value().([]float32)
		if len(allActionsProbs) != ab.LenActions() {
			log.Panicf("Total probabilities returned was %d, wanted %d",
				len(allActionsProbs), ab.LenActions())
		}
		for _, req := range ab.requests {
			req.actionsProbs = allActionsProbs[:req.LenActions()]
			allActionsProbs = allActionsProbs[req.LenActions():]
		}
	}

	// Signal that request has been fulfilled.
	for _, req := range ab.requests {
		close(req.done)
	}
}

func (s *Scorer) autoBatchDispatcher() {
	klog.V(1).Infof("Started AutoBatch dispatcher for [%s].", s)
	var ab *AutoBatch
	for req := range s.autoBatchChan {
		if req != onBatchSizeUpdate {
			if ab == nil {
				ab = s.newAutoBatch()
			}
			ab.Append(req)
			klog.V(3).Info("Received scoring request.")
		} else {
			klog.V(1).Infof("[%s] batch size changed to %d", s, s.autoBatchSize)
		}
		if ab != nil && ab.Len() >= s.autoBatchSize {
			go s.autoBatchScoreAndDeliver(ab)
			ab = nil
		}
	}
}
