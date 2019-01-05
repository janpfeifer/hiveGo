package tensorflow

import (
	"log"
	"math/rand"

	. "github.com/janpfeifer/hiveGo/state"

	"github.com/golang/glog"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// randomMiniBatches will split up the flatFeaturesCollection into many
// random mini-batches of the given size (resampling with no replacement).
//
// The data itself is not copied, only the slices, so the original
// flatFeturesCollection passed must be preserved.
//
// If there is not enough examples to fill the last minibatch (that is
// if the number of examples is not divisible by batchSize), the
// last minibatch is discarded.
func (fc *flatFeaturesCollection) randomMiniBatches(batchSize int) (fcs []*flatFeaturesCollection) {
	numBatches := fc.Len() / batchSize
	fcs = make([]*flatFeaturesCollection, numBatches)

	order := make([]int, fc.Len())
	for ii := 0; ii < fc.Len(); ii++ {
		order[ii] = ii
	}
	for ii := 0; ii < fc.Len(); ii++ {
		jj := rand.Intn(fc.Len())
		order[ii], order[jj] = order[jj], order[ii]
	}

	// Get reverse order so data is processed serially (hopefully
	// better data locality).
	srcIdxToBatchNum := make([]int, fc.Len())
	for ii := 0; ii < fc.Len(); ii++ {
		batchNum := ii / batchSize
		if batchNum >= numBatches {
			batchNum = -1
		}
		srcIdx := order[ii]
		srcIdxToBatchNum[srcIdx] = batchNum
	}

	for srcIdx, srcActionIdx := 0, 0; srcIdx < fc.Len(); srcIdx++ {
		batchNum := srcIdxToBatchNum[srcIdx]
		if batchNum < 0 {
			// Skip corresponding actions, even if board was not used.
			for ; srcActionIdx < fc.totalNumActions && fc.actionsBoardIndices[srcActionIdx] == int64(srcIdx); srcActionIdx++ {
			}
			continue
		}

		// Get or allocate new flatFeaturesCollection for the corresponding batch.
		batchFC := fcs[batchNum]
		if batchFC == nil {
			batchFC = &flatFeaturesCollection{
				boardFeatures: make([][]float32, 0, batchSize),
				boardLabels:   make([]float32, 0, batchSize),
			}
			fcs[batchNum] = batchFC
		}

		// Append board features.
		batchFCIdx := batchFC.Len()
		batchFC.boardFeatures = append(batchFC.boardFeatures, fc.boardFeatures[srcIdx])
		batchFC.fullBoardFeatures = append(batchFC.fullBoardFeatures, fc.fullBoardFeatures[srcIdx])
		batchFC.boardLabels = append(batchFC.boardLabels, fc.boardLabels[srcIdx])

		// Append action features.
		srcActionStart := srcActionIdx
		for ; srcActionIdx < fc.totalNumActions && fc.actionsBoardIndices[srcActionIdx] == int64(srcIdx); srcActionIdx++ {
			batchFC.actionsBoardIndices = append(batchFC.actionsBoardIndices, int64(batchFCIdx))
		}
		batchFC.actionsIsMove = append(batchFC.actionsIsMove, fc.actionsIsMove[srcActionStart:srcActionIdx]...)
		batchFC.actionsSrcPositions = append(batchFC.actionsSrcPositions, fc.actionsSrcPositions[srcActionStart:srcActionIdx]...)
		batchFC.actionsTgtPositions = append(batchFC.actionsTgtPositions, fc.actionsTgtPositions[srcActionStart:srcActionIdx]...)
		batchFC.actionsPieces = append(batchFC.actionsPieces, fc.actionsPieces[srcActionStart:srcActionIdx]...)
		batchFC.actionsLabels = append(batchFC.actionsLabels, fc.actionsLabels[srcActionStart:srcActionIdx]...)
		batchFC.totalNumActions += (srcActionIdx - srcActionStart)
	}

	// Sanity checks.
	for num, batchFC := range fcs {
		if batchFC.Len() != batchSize {
			log.Panicf("Minibatch %d has %d boards, wanted %d (batchSize)", num, batchFC.Len(), batchSize)
		}
	}
	return
}

func (s *Scorer) buildFeedsForLearning(batch *flatFeaturesCollection, scoreActions bool) (
	feeds map[tf.Output]*tf.Tensor) {
	feeds = s.buildFeeds(batch, scoreActions)

	// Feed also the labels.
	feeds[s.BoardLabels] = mustTensor(batch.boardLabels)
	if scoreActions {
		feeds[s.ActionsLabels] = mustTensor(batch.actionsLabels)
	}
	if s.IsTraining.Op != nil {
		feeds[s.IsTraining] = mustTensor(true)
	}
	return
}

// learnOneStep will run 'trainOp' (if train==true) and return the loss on
// the minibatch set up on feeds.
func (s *Scorer) learnOneMiniBatch(feeds map[tf.Output]*tf.Tensor, train, scoreActions bool) (
	totalLoss, boardLoss, actionsLoss float32) {
	fetches := []tf.Output{s.TotalLoss}
	if scoreActions {
		fetches = append(fetches, s.BoardLosses)
		fetches = append(fetches, s.ActionsLosses)
	}
	var ops []*tf.Operation
	if train {
		ops = append(ops, s.TrainOp)
	} else {
		if s.IsTraining.Op != nil {
			// If only evaluating, set IsTraining to false (which disables dropout).
			feeds[s.IsTraining] = mustTensor(false)
		}
	}
	results, err := s.sessionPool[0].Run(feeds, fetches, ops)
	if err != nil {
		log.Panicf("TensorFlow trainOp failed: %v", err)
	}
	totalLoss = results[0].Value().(float32)
	if scoreActions {
		boardLoss = results[1].Value().(float32)
		actionsLoss = results[2].Value().(float32)
	} else {
		boardLoss = 0
		actionsLoss = 0
	}
	return
}

// Learn will train the tensorflow model for the given labels and actionLabels.
// learningRate is ignored, instead it is read as a standard parameter.
func (s *Scorer) Learn(
	boards []*Board, boardLabels []float32, actionsLabels [][]float32,
	_ float32, epochs int, perStepCallback func()) (
	loss, boardLoss, actionsLoss float32) {
	s.parseFlagParamsFile()
	if len(boards) == 0 {
		log.Panicf("Received empty list of boards to learn.")
	}
	if len(s.sessionPool) > 1 {
		log.Panicf("Using SessionPool doesn't support training. You probably should use sessionPoolSize=1 in this case.")
	}
	scoreActions := s.IsActionsClassifier() && actionsLabels != nil
	fc := s.buildFeatures(boards, scoreActions)
	fc.boardLabels = boardLabels
	if scoreActions {
		for boardIdx, a := range actionsLabels {
			if boards[boardIdx].NumActions() > 1 {
				if len(a) != boards[boardIdx].NumActions() {
					log.Panicf("%d actionsLabeles given to board, but there are %d actions", len(a), boards[boardIdx].NumActions())
				}
			}
			fc.actionsLabels = append(fc.actionsLabels, a...)
		}
		if len(fc.actionsLabels) != fc.totalNumActions {
			log.Panicf("%d actions in fc, but only %d labels given.", fc.totalNumActions, len(fc.actionsLabels))
		}
	}

	if *Flag_learnBatchSize == 0 || len(boards) <= *Flag_learnBatchSize {
		for epoch := 0; epoch < epochs || epoch == 0; epoch++ {
			feeds := s.buildFeedsForLearning(fc, scoreActions)
			loss, boardLoss, actionsLoss = s.learnOneMiniBatch(feeds, epochs > 0, scoreActions)
			boardLoss /= float32(len(boards))
			actionsLoss /= float32(len(boards))
			if epochs > 0 && perStepCallback != nil {
				perStepCallback()
			}
			glog.V(1).Infof("Loss after epoch: total=%g, board=%g, actions=%g",
				loss, boardLoss, actionsLoss)
		}
		return
	}

	// Write batches in one goroutine.
	batchesChan := make(chan *flatFeaturesCollection, 10)
	go func() {
		first := true
		for epoch := 0; epoch < epochs || epoch == 0; epoch++ {
			miniBatches := fc.randomMiniBatches(*Flag_learnBatchSize)
			if first {
				glog.V(1).Infof("Learn with %d mini-batches of size %d", len(miniBatches), *Flag_learnBatchSize)
				first = false
			}
			for _, batch := range miniBatches {
				batchesChan <- batch
			}
			batchesChan <- nil
		}
		close(batchesChan)
	}()

	// Write feeds in a separate goroutine.
	feedsChan := make(chan map[tf.Output]*tf.Tensor, 10)
	go func() {
		for batch := range batchesChan {
			if batch == nil {
				feedsChan <- nil
				continue
			}
			feedsChan <- s.buildFeedsForLearning(batch, scoreActions)
		}
		close(feedsChan)
	}()

	// Learn from pre-generate feeds:
	countExamples, countEpoch := 0, 0
	var averageLoss, averageBoardLoss, averageActionsLoss float32
	for feeds := range feedsChan {
		if feeds == nil {
			averageLoss /= float32(countExamples)
			averageBoardLoss /= float32(countExamples)
			averageActionsLoss /= float32(countExamples)
			glog.V(1).Infof("Loss after epoch %d: total=%g, board=%g, actions=%g",
				countEpoch, averageLoss,
				averageBoardLoss/float32(*Flag_learnBatchSize),
				averageActionsLoss/float32(*Flag_learnBatchSize))
			countEpoch++
			countExamples = 0
			if epochs > 0 && perStepCallback != nil {
				perStepCallback()
			}
			continue
		}
		if countExamples == 0 {
			// At the start of a new batch, reset average losses.
			averageLoss, averageBoardLoss, averageActionsLoss = 0, 0, 0
		}
		totalLoss, boardLoss, actionsLoss := s.learnOneMiniBatch(feeds, epochs > 0, scoreActions)
		averageLoss += totalLoss
		averageBoardLoss += boardLoss
		averageActionsLoss += actionsLoss
		countExamples++
	}
	return averageLoss, averageBoardLoss / float32(*Flag_learnBatchSize), averageActionsLoss / float32(*Flag_learnBatchSize)
}
