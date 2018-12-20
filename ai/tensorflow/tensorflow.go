package tensorflow

// Google's support for Tensorflow in Go is still lacking. To get the Tensorlow
// protobuffers needed compiled for go, do the following:
//
// 1) Install the Go Proto tool support. Details here:
//
// https://developers.google.com/protocol-buffers/docs/reference/go-generated
//
// I did the following:
//      go get github.com/golang/protobuf/proto
//      go get github.com/golang/protobuf/protoc-gen-go
//
// Have protoc installed (sudo apt install protobuf-compiler)
//
// 2) Get Tensorflow proto definitions (.proto files):
//
//      (From a directory called ${REPOS})
//      git clone git@github.com:tensorflow/tensorflow.git
//
// 3) Compile protos to Go:
//      ${REPOS} -> where you got the tensorflow sources in (2)
//      ${GOSRC} -> your primary location of Go packages, typically the first entry in GOPATH.
//      for ii in config.proto debug.proto cluster.proto rewriter_config.proto ; do
//        protoc --proto_path=${REPOS}/tensorflow --go_out=${GOSRC}/src \
//          ${REPOS}/tensorflow/tensorflow/core/protobuf/${ii}
//      done
//      protoc --proto_path=${REPOS}/tensorflow --go_out=${GOSRC}/src \
//          ${REPOS}/tensorflow/tensorflow/core/framework/*.proto
//
//      You can convert other protos as needed -- yes, unfortunately I only need config.proto
//      but had to manually track the dependencies ... :(
import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"sync"

	"github.com/golang/protobuf/proto"
	tfconfig "github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf"

	"github.com/golang/glog"
	"github.com/janpfeifer/hiveGo/ai"
	"github.com/janpfeifer/hiveGo/ai/players"
	. "github.com/janpfeifer/hiveGo/state"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Set this to true to force to use CPU, even when GPU is avaialble.
var CpuOnly = false

const (
	INTER_OP_PARALLELISM       = 4
	INTRA_OP_PARALLELISM       = 4
	GPU_MEMORY_FRACTION_TO_USE = 0.3
)

var flag_learnBatchSize = flag.Int("tf_batch_size", 0,
	"Batch size when learning: this is the number of boards, not actions. There is usually 100/1 ratio of "+
		"actions per board. Examples are shuffled before being batched. 0 means no batching.")

type Scorer struct {
	Basename            string
	graph               *tf.Graph
	sessionPool         []*tf.Session
	sessionTurn         int // Rotate among the sessions from the pool.
	isActionsClassifier bool
	mu                  sync.Mutex

	// Auto-batching waits for some requests to arrive before actually calling tensorflow.
	// The idea being to make better CPU/GPU utilization.
	autoBatchSize int
	autoBatchChan chan *AutoBatchRequest

	BoardFeatures, BoardLabels    tf.Output
	FullBoard                     tf.Output
	BoardPredictions, BoardLosses tf.Output

	ActionsBoardIndices, ActionsIsMove       tf.Output
	ActionsSrcPositions, ActionsTgtPositions tf.Output
	ActionsPieces, ActionsLabels             tf.Output
	ActionsPredictions, ActionsLosses        tf.Output

	IsTraining, LearningRate, CheckpointFile, TotalLoss tf.Output
	SelfSupervision                                     tf.Output
	InitOp, TrainOp, SaveOp, RestoreOp                  *tf.Operation

	version int
	// Uses the number of input features used.
}

// Data used for parsing of player options.
type ParsingData struct {
	UseTensorFlow, ForceCPU bool
	SessionPoolSize         int
}

func NewParsingData() (data interface{}) {
	return &ParsingData{SessionPoolSize: 1}
}

func FinalizeParsing(data interface{}, player *players.SearcherScorerPlayer) {
	d := data.(*ParsingData)
	if d.UseTensorFlow {
		player.Learner = New(player.ModelFile, d.SessionPoolSize, d.ForceCPU)
		player.Scorer = player.Learner
	}
}

func ParseParam(data interface{}, key, value string) {
	d := data.(*ParsingData)
	if key == "tf" {
		d.UseTensorFlow = true
	} else if key == "tf_cpu" {
		d.ForceCPU = true
	} else if key == "tf_session_pool_size" {
		var err error
		d.SessionPoolSize, err = strconv.Atoi(value)
		if err != nil {
			log.Panicf("Invalid parameter tf_session_pool_size=%s: %v", value, err)
		}
		if d.SessionPoolSize < 1 {
			log.Panicf("Invalid parameter tf_session_pool_size=%s, it must be > 0", value)
		}
	} else {
		log.Panicf("Unknown parameter '%s=%s' passed to tensorflow module.", key, value)
	}
}

func init() {
	players.RegisterPlayerParameter("tf", "tf", NewParsingData, ParseParam, FinalizeParsing)
	players.RegisterPlayerParameter("tf", "tf_cpu", NewParsingData, ParseParam, FinalizeParsing)
	players.RegisterPlayerParameter("tf", "tf_session_pool_size", NewParsingData, ParseParam, FinalizeParsing)
}

var dataTypeMap = map[tf.DataType]string{
	tf.Float:  "tf.float32",
	tf.Double: "tf.float64",
	tf.Int32:  "tf.int32",
	tf.Int64:  "tf.int64",
	tf.String: "tf.string",
}

func dataType(t tf.Output) string {
	dt := t.DataType()
	str, ok := dataTypeMap[dt]
	if !ok {
		return fmt.Sprintf("type_%d?", int(dt))
	}
	return str
}

// New creates a new Scorer by reading model's graph `basename`.pb,
// and checkpoints from `basename`.checkpoint
func New(basename string, sessionPoolSize int, forceCPU bool) *Scorer {
	// Load graph definition (as bytes) and import into current graph.
	graphDefFilename := fmt.Sprintf("%s.pb", basename)
	graphDef, err := ioutil.ReadFile(graphDefFilename)
	if err != nil {
		log.Panicf("Failed to read %q: %v", graphDefFilename, err)
	}

	// Create the one graph and sessions we will use all time.
	graph := tf.NewGraph()

	if err = graph.Import(graphDef, ""); err != nil {
		log.Fatal("Invalid GraphDef? read from %s: %v", graphDefFilename, err)
	}

	absBasename, err := filepath.Abs(basename)
	if err != nil {
		log.Panicf("Unknown absolute path for %s: %v", basename, err)
	}

	t0 := func(tensorName string) (to tf.Output) {
		op := graph.Operation(tensorName)
		if op == nil {
			log.Fatalf("Failed to find tensor [%s]", tensorName)
		}
		return op.Output(0)
	}
	t0opt := func(tensorName string) (to tf.Output) {
		op := graph.Operation(tensorName)
		if op == nil {
			return
		}
		return op.Output(0)
	}

	op := func(tensorName string) *tf.Operation {
		return graph.Operation(tensorName)
	}

	s := &Scorer{
		Basename:      absBasename,
		graph:         graph,
		sessionPool:   createSessionPool(graph, sessionPoolSize, forceCPU),
		autoBatchSize: 1,
		autoBatchChan: make(chan *AutoBatchRequest),

		// Board tensors.
		BoardFeatures:    t0("board_features"),
		BoardLabels:      t0("board_labels"),
		FullBoard:        t0opt("full_board"),
		BoardPredictions: t0("board_predictions"),
		BoardLosses:      t0("board_losses"),

		// Actions inputs.
		ActionsBoardIndices: t0opt("actions_board_indices"),
		ActionsIsMove:       t0opt("actions_is_move"),
		ActionsSrcPositions: t0opt("actions_src_positions"),
		ActionsTgtPositions: t0opt("actions_tgt_positions"),
		ActionsPieces:       t0opt("actions_pieces"),
		ActionsLabels:       t0opt("actions_labels"),

		// Actions related output.
		ActionsPredictions: t0opt("actions_predictions"),
		ActionsLosses:      t0opt("actions_losses"),

		// Global parameters.
		IsTraining:      t0opt("is_training"),
		LearningRate:    t0("learning_rate"),
		SelfSupervision: t0opt("self_supervision"),
		CheckpointFile:  t0("save/Const"),
		TotalLoss:       t0("mean_loss"),

		// Ops.
		InitOp:    op("init"),
		TrainOp:   op("train"),
		SaveOp:    op("save/control_dependency"),
		RestoreOp: op("save/restore_all"),
	}

	// Model must have all actions tensors to be considered actions classifier.
	s.isActionsClassifier =
		s.ActionsBoardIndices.Op != nil && s.ActionsIsMove.Op != nil &&
			s.ActionsSrcPositions.Op != nil && s.ActionsTgtPositions.Op != nil &&
			s.ActionsPieces.Op != nil && s.ActionsLabels.Op != nil &&
			s.ActionsPredictions.Op != nil && s.ActionsLosses.Op != nil
	if !s.isActionsClassifier {
		glog.Infof("%s can not be used for actions classification.", s)
	}

	// Set version to the size of the input.
	s.version = int(s.BoardFeatures.Shape().Size(1))
	glog.V(1).Infof("TensorFlow model's version=%d", s.version)

	// Either restore or initialize the network.
	cpIndex, _ := s.CheckpointFiles()
	if _, err := os.Stat(cpIndex); err == nil {
		glog.Infof("Loading model from %s", s.CheckpointBase())
		err = s.Restore()
		if err != nil {
			log.Panicf("Failed to load checkpoint from file %s: %v", s.CheckpointBase(), err)
		}
	} else if os.IsNotExist(err) {
		glog.Infof("Initializing model randomly, since %s not found", s.CheckpointBase())
		err = s.Init()
		if err != nil {
			log.Panicf("Failed to initialize model: %v", err)
		}
	} else {
		log.Panicf("Cannot checkpoint file %s: %v", s.CheckpointBase(), err)
	}

	go s.autoBatchDispatcher()
	return s
}

func (s *Scorer) IsActionsClassifier() bool {
	return s.isActionsClassifier
}

func (s *Scorer) HasFullBoard() bool {
	return s.FullBoard.Op != nil
}

func createSessionPool(graph *tf.Graph, size int, forceCPU bool) (sessions []*tf.Session) {
	gpuMemFractionLeft := GPU_MEMORY_FRACTION_TO_USE
	for ii := 0; ii < size; ii++ {
		sessionOptions := &tf.SessionOptions{}
		var config tfconfig.ConfigProto
		if forceCPU || CpuOnly {
			// TODO this doesn't work .... :(
			// Instead use:
			//    export CUDA_VISIBLE_DEVICES=-1
			// Before starting the program.
			config.DeviceCount = map[string]int32{"GPU": 0}
		} else {
			config.GpuOptions = &tfconfig.GPUOptions{}
			config.GpuOptions.PerProcessGpuMemoryFraction = gpuMemFractionLeft / float64(size-ii)
			gpuMemFractionLeft -= config.GpuOptions.PerProcessGpuMemoryFraction
		}
		config.InterOpParallelismThreads = INTER_OP_PARALLELISM
		config.IntraOpParallelismThreads = INTRA_OP_PARALLELISM
		data, err := proto.Marshal(&config)
		if err != nil {
			log.Panicf("Failed to serialize tf.ConfigProto: %v", err)
		}
		sessionOptions.Config = data
		sess, err := tf.NewSession(graph, sessionOptions)
		if err != nil {
			log.Panicf("Failed to create tensorflow session: %v", err)
		}
		if ii == 0 {
			devices, _ := sess.ListDevices()
			glog.Infof("List of available devices: %v", devices)
		}
		sessions = append(sessions, sess)
	}
	return
}

func (s *Scorer) String() string {
	return fmt.Sprintf("TensorFlow model in '%s'", s.Basename)
}

func (s *Scorer) NextSession() (sess *tf.Session) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess = s.sessionPool[s.sessionTurn]
	s.sessionTurn = (s.sessionTurn + 1) % len(s.sessionPool)
	return
}

func (s *Scorer) CheckpointBase() string {
	return fmt.Sprintf("%s.checkpoint", s.Basename)
}

func (s *Scorer) CheckpointFiles() (string, string) {
	return fmt.Sprintf("%s.checkpoint.index", s.Basename), fmt.Sprintf("%s.checkpoint.data-00000-of-00001", s.Basename)

}

func (s *Scorer) Restore() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, sess := range s.sessionPool {
		t, err := tf.NewTensor(s.CheckpointBase())
		if err != nil {
			log.Panicf("Failed to create tensor: %v", err)
		}
		feeds := map[tf.Output]*tf.Tensor{
			s.CheckpointFile: t,
		}
		_, err = sess.Run(feeds, nil, []*tf.Operation{s.RestoreOp})
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *Scorer) Init() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, sess := range s.sessionPool {
		_, err := sess.Run(nil, nil, []*tf.Operation{s.InitOp})
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *Scorer) Version() int {
	return s.version
}

func (s *Scorer) Score(b *Board, scoreActions bool) (score float32, actionProbs []float32) {
	if s.autoBatchSize > 0 {
		// Use auto-batching
		return s.scoreAutoBatch(b, scoreActions)
	}
	boards := []*Board{b}
	scores, actionProbsBatch := s.BatchScore(boards, scoreActions)
	return scores[0], actionProbsBatch[0]
}

// Quick utility to create a tensor out of value. Dies if there is an error.
func mustTensor(value interface{}) *tf.Tensor {
	tensor, err := tf.NewTensor(value)
	if err != nil {
		log.Panicf("Cannot convert to tensor: %v", err)
	}
	return tensor
}

// flatFeaturesCollection aggregate the features for several examples. It
// can also hold the labels.
type flatFeaturesCollection struct {
	boardFeatures     [][]float32
	fullBoardFeatures [][][][]float32 // [batch, height, width, depth]
	boardLabels       []float32

	// Labels
	actionsBoardIndices []int64 // Go tensorflow implementation is broken for int32.
	actionsIsMove       []bool
	actionsSrcPositions [][2]int64
	actionsTgtPositions [][2]int64
	actionsPieces       [][NUM_PIECE_TYPES]float32
	actionsLabels       []float32
	totalNumActions     int
}

func (s *Scorer) buildFeatures(boards []*Board, scoreActions bool) (fc *flatFeaturesCollection) {
	// Initialize Go objects, that need to be copied to tensors.
	fc = &flatFeaturesCollection{}
	fc.boardFeatures = make([][]float32, len(boards))
	var batchWidth, batchHeight int
	if s.HasFullBoard() {
		fc.fullBoardFeatures = make([][][][]float32, len(boards))
		batchWidth, batchHeight = ai.SuggestedFullBoardWidth, ai.SuggestedFullBoardHeight
		for _, b := range boards {
			w, h := ai.FullBoardDimensions(b)
			if w > batchWidth {
				batchWidth = w
			}
			if h > batchHeight {
				batchHeight = h
			}
		}
	}

	if scoreActions {
		for _, board := range boards {
			fc.totalNumActions += board.NumActions()
		}
		fc.actionsBoardIndices = make([]int64, fc.totalNumActions) // Go tensorflow implementation is broken for int32.
		fc.actionsIsMove = make([]bool, fc.totalNumActions)
		fc.actionsSrcPositions = make([][2]int64, fc.totalNumActions)
		fc.actionsTgtPositions = make([][2]int64, fc.totalNumActions)
		fc.actionsPieces = make([][NUM_PIECE_TYPES]float32, fc.totalNumActions)
	}

	// Generate features in Go slices.
	actionIdx := 0
	for boardIdx, board := range boards {
		fc.boardFeatures[boardIdx] = ai.FeatureVector(board, s.version)
		if s.HasFullBoard() {
			fc.fullBoardFeatures[boardIdx] = ai.MakeFullBoardFeatures(board, batchWidth, batchHeight)
		}
		if scoreActions {
			for _, action := range board.Derived.Actions {
				fc.actionsBoardIndices[actionIdx] = int64(boardIdx)
				fc.actionsIsMove[actionIdx] = action.Move
				fc.actionsSrcPositions[actionIdx] = ai.PosToFullBoardPosition(board, action.SourcePos)
				fc.actionsTgtPositions[actionIdx] = ai.PosToFullBoardPosition(board, action.TargetPos)
				fc.actionsPieces[actionIdx][int(action.Piece)-1] = 1
				actionIdx++
			}
		}
	}
	return
}

func (fc *flatFeaturesCollection) Len() int {
	return len(fc.boardFeatures)
}

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

func (s *Scorer) buildFeeds(fc *flatFeaturesCollection, scoreActions bool) (feeds map[tf.Output]*tf.Tensor) {
	// Convert Go slices to tensors.
	if scoreActions {
		feeds = map[tf.Output]*tf.Tensor{
			s.BoardFeatures:       mustTensor(fc.boardFeatures),
			s.ActionsBoardIndices: mustTensor(fc.actionsBoardIndices),
			s.ActionsIsMove:       mustTensor(fc.actionsIsMove),
			s.ActionsSrcPositions: mustTensor(fc.actionsSrcPositions),
			s.ActionsTgtPositions: mustTensor(fc.actionsTgtPositions),
			s.ActionsPieces:       mustTensor(fc.actionsPieces),
		}
	} else {
		feeds = map[tf.Output]*tf.Tensor{
			s.BoardFeatures: mustTensor(fc.boardFeatures),
		}
	}
	if s.HasFullBoard() {
		if glog.V(2) {
			f := fc.fullBoardFeatures
			shape := []int{len(f), len(f[0]), len(f[0][0]), len(f[0][0][0])}
			glog.Infof("fullBoardFeatures shape: %v", shape)
			if scoreActions {
				glog.Infof("ActionsBoardIndices shape: %v", []int{len(fc.actionsBoardIndices)})
				glog.Infof("ActionsIsMove shape: %v", []int{len(fc.actionsIsMove)})
				glog.Infof("ActionsActionsPieces shape: %v", []int{len(fc.actionsPieces), len(fc.actionsPieces[0])})
			}
		}
		feeds[s.FullBoard] = mustTensor(fc.fullBoardFeatures)
	}
	return
}

func (s *Scorer) BatchScore(boards []*Board, scoreActions bool) (scores []float32, actionProbsBatch [][]float32) {
	if len(boards) == 0 {
		log.Panicf("Received empty list of boards to score.")
	}

	// Build feeds to TF model.
	fc := s.buildFeatures(boards, scoreActions)
	feeds := s.buildFeeds(fc, scoreActions)
	if s.IsTraining.Op != nil {
		feeds[s.IsTraining] = mustTensor(false)
	}
	if s.HasFullBoard() {
		feeds[s.FullBoard] = mustTensor(fc.fullBoardFeatures)
	}

	fetches := []tf.Output{s.BoardPredictions}
	if scoreActions && fc.totalNumActions > 0 {
		fetches = append(fetches, s.ActionsPredictions)
	}

	// Evaluate: at most one evaluation at a same time.
	if glog.V(3) {
		glog.V(3).Infof("Feeded tensors: ")
		for to, tensor := range feeds {
			glog.V(3).Infof("\t%s: %v", to.Op.Name(), tensor.Shape())
		}
	}

	sess := s.NextSession()
	results, err := sess.Run(feeds, fetches, nil)
	if err != nil {
		log.Panicf("Prediction failed: %v", err)
	}

	// Copy over resulting tensors.
	scores = results[0].Value().([]float32)
	if len(scores) != len(boards) {
		log.Panicf("Expected %d scores (=number of boards given), got %d",
			len(boards), len(scores))
	}

	actionProbsBatch = make([][]float32, len(boards))
	if fc.totalNumActions > 0 {
		allActionsProbs := results[1].Value().([]float32)
		if len(allActionsProbs) != fc.totalNumActions {
			log.Panicf("Total probabilities returned was %d, wanted %d",
				len(allActionsProbs), fc.totalNumActions)
		}
		if len(allActionsProbs) != fc.totalNumActions {
			log.Panicf("Expected %d actions (from %d boards), got %d",
				fc.totalNumActions, len(boards), len(allActionsProbs))
		}
		for boardIdx, board := range boards {
			actionProbsBatch[boardIdx] = allActionsProbs[:board.NumActions()]
			allActionsProbs = allActionsProbs[len(board.Derived.Actions):]
			if len(actionProbsBatch[boardIdx]) != board.NumActions() {
				log.Panicf("Got %d probabilities for %d actions!?", len(actionProbsBatch[boardIdx]),
					board.NumActions())
			}
		}
	}
	return
}

func (s *Scorer) learnOneMiniBatch(
	batch *flatFeaturesCollection, learningRate float32, steps int, scoreActions bool) (
	totalLoss, boardLoss, actionsLoss float32) {
	feeds := s.buildFeeds(batch, scoreActions)

	// Feed also the labels.
	feeds[s.BoardLabels] = mustTensor(batch.boardLabels)
	if scoreActions {
		feeds[s.ActionsLabels] = mustTensor(batch.actionsLabels)
	}
	feeds[s.LearningRate] = mustTensor(learningRate)
	if s.IsTraining.Op != nil {
		feeds[s.IsTraining] = mustTensor(true)
	}
	if s.SelfSupervision.Op != nil {
		feeds[s.SelfSupervision] = mustTensor(float32(0))
	}

	// Loop over steps.
	for step := 0; step < steps || step == 0; step++ {
		var fetches []tf.Output
		if step == steps-1 || steps == 0 {
			fetches = append(fetches, s.TotalLoss)
			if scoreActions {
				fetches = append(fetches, s.BoardLosses)
				fetches = append(fetches, s.ActionsLosses)
			}
		}
		var ops []*tf.Operation
		if steps > 0 {
			ops = append(ops, s.TrainOp)
		}
		results, err := s.sessionPool[0].Run(feeds, fetches, ops)
		if err != nil {
			log.Panicf("TensorFlow trainOp failed: %v", err)
		}
		if step == steps-1 || steps == 0 {
			totalLoss = results[0].Value().(float32)
			if scoreActions {
				boardLoss = results[1].Value().(float32)
				actionsLoss = results[2].Value().(float32)
			}
		}
	}
	return
}

func (s *Scorer) Learn(
	boards []*Board, boardLabels []float32, actionsLabels [][]float32,
	learningRate float32, steps int, perStepCallback func()) (loss float32) {
	if len(boards) == 0 {
		log.Panicf("Received empty list of boards to learn.")
	}
	if len(s.sessionPool) > 1 {
		log.Panicf("Using SessionPool doesn't support training. You probably should use sessionPoolSize=1 in this case.")
	}
	scoreActions := s.IsActionsClassifier() && actionsLabels != nil
	fc := s.buildFeatures(boards, scoreActions)
	fc.boardLabels = boardLabels
	for boardIdx := range fc.boardLabels {
		fc.boardLabels[boardIdx] = ai.SigmoidTo10(fc.boardLabels[boardIdx])
	}
	if scoreActions {
		for boardIdx, a := range actionsLabels {
			if len(a) != boards[boardIdx].NumActions() {
				log.Panicf("%d actionsLabeles given to board, but there are %d actions", len(a), boards[boardIdx].NumActions())
			}
			fc.actionsLabels = append(fc.actionsLabels, a...)
		}
		if len(fc.actionsLabels) != fc.totalNumActions {
			log.Panicf("%d actions in fc, but only %d labels given.", fc.totalNumActions, len(fc.actionsLabels))
		}
	}

	if *flag_learnBatchSize == 0 || len(boards) < *flag_learnBatchSize {
		// Only one batch case:
		miniBatchSteps := 1
		if steps == 0 {
			miniBatchSteps = 0
		}
		for step := 0; step < steps || step == 0; step++ {
			totalLoss, boardLoss, actionsLoss := s.learnOneMiniBatch(fc, learningRate, miniBatchSteps, scoreActions)
			if steps > 0 && perStepCallback != nil {
				perStepCallback()
			}
			glog.V(1).Infof("Loss after epoch: total=%g, board=%g, actions=%g",
				totalLoss, boardLoss, actionsLoss)
			loss = totalLoss
		}
		return
	}

	var averageLoss, averageBoardLoss, averageActionsLoss float32
	for step := 0; step < steps || step == 0; step++ {
		miniBatches := fc.randomMiniBatches(*flag_learnBatchSize)
		glog.V(1).Infof("Learn with %d mini-batches of size %d, epoch %d", len(miniBatches), *flag_learnBatchSize, step)
		averageLoss, averageBoardLoss, averageActionsLoss = 0, 0, 0
		miniBatchSteps := 1
		if steps == 0 {
			miniBatchSteps = 0
		}
		for _, batch := range miniBatches {
			totalLoss, boardLoss, actionsLoss := s.learnOneMiniBatch(batch, learningRate, miniBatchSteps, scoreActions)
			averageLoss += totalLoss
			averageBoardLoss += boardLoss
			averageActionsLoss += actionsLoss
		}
		averageLoss /= float32(len(miniBatches))
		averageBoardLoss /= float32(len(miniBatches))
		averageActionsLoss /= float32(len(miniBatches))
		glog.V(1).Infof("Loss after epoch: total=%g, board=%g, actions=%g",
			averageLoss, averageBoardLoss, averageActionsLoss)
		if steps > 0 && perStepCallback != nil {
			perStepCallback()
		}
	}
	return averageLoss
}

func (s *Scorer) Save() {
	if len(s.sessionPool) > 1 {
		log.Panicf("SessionPool doesn't support saving. You probably should use sessionPoolSize=1 in this case.")
	}

	// Backup previous checkpoint.
	index, data := s.CheckpointFiles()
	if _, err := os.Stat(index); err == nil {
		if err := os.Rename(index, index+"~"); err != nil {
			glog.Errorf("Failed to backup %s to %s~: %v", index, index, err)
		}
		if err := os.Rename(data, data+"~"); err != nil {
			glog.Errorf("Failed to backup %s to %s~: %v", data, data, err)
		}
	}

	t, err := tf.NewTensor(s.CheckpointBase())
	if err != nil {
		log.Panicf("Failed to create tensor: %v", err)
	}
	feeds := map[tf.Output]*tf.Tensor{s.CheckpointFile: t}
	if _, err := s.sessionPool[0].Run(feeds, nil, []*tf.Operation{s.SaveOp}); err != nil {
		log.Panicf("Failed to checkpoint (save) file to %s: %v", s.CheckpointBase(), err)
	}
}

type AutoBatchRequest struct {
	// Boards
	boardFeatures     []float32
	fullBoardFeatures [][][]float32 // [batch, height, width, depth]

	// Actions
	actionsIsMove       []bool
	actionsSrcPositions [][2]int64
	actionsTgtPositions [][2]int64
	actionsPieces       [][NUM_PIECE_TYPES]float32

	// Channel is closed when done.
	done chan bool

	// Results
	score        float32
	actionsProbs []float32
}

func (s *Scorer) newAutoBatchRequest(b *Board, scoreActions bool) (req *AutoBatchRequest) {
	req = &AutoBatchRequest{
		boardFeatures: ai.FeatureVector(b, s.version),
		done:          make(chan bool),
	}
	if s.HasFullBoard() {
		req.fullBoardFeatures = ai.MakeFullBoardFeatures(b, ai.SuggestedFullBoardWidth, ai.SuggestedFullBoardHeight)
	}
	scoreActions = scoreActions && s.IsActionsClassifier()
	if scoreActions && b.NumActions() > 0 {
		req.actionsIsMove = make([]bool, b.NumActions())
		req.actionsSrcPositions = make([][2]int64, b.NumActions())
		req.actionsTgtPositions = make([][2]int64, b.NumActions())
		req.actionsPieces = make([][NUM_PIECE_TYPES]float32, b.NumActions())
		for actionIdx, action := range b.Derived.Actions {
			req.actionsIsMove[actionIdx] = action.Move
			req.actionsSrcPositions[actionIdx] = ai.PosToFullBoardPosition(b, action.SourcePos)
			req.actionsTgtPositions[actionIdx] = ai.PosToFullBoardPosition(b, action.TargetPos)
			req.actionsPieces[actionIdx][int(action.Piece)-1] = 1
		}
	}
	return
}

func (req *AutoBatchRequest) LenActions() int { return len(req.actionsIsMove) }

func (s *Scorer) scoreAutoBatch(b *Board, scoreActions bool) (score float32, actionsProbs []float32) {
	// Send request and wait for it to be processed.
	req := s.newAutoBatchRequest(b, scoreActions)
	glog.V(3).Info("Sending request", s)
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
	actionsPieces       [][NUM_PIECE_TYPES]float32
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
		ab.actionsPieces = make([][NUM_PIECE_TYPES]float32, 0, maxActions)
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
	if s.SelfSupervision.Op != nil {
		feeds[s.SelfSupervision] = mustTensor(float32(0))
	}

	fetches := []tf.Output{s.BoardPredictions}
	if s.IsActionsClassifier() && ab.LenActions() > 0 {
		fetches = append(fetches, s.ActionsPredictions)
	}
	// Evaluate: at most one evaluation at a same time.
	if glog.V(3) {
		glog.V(3).Infof("Tensors fed: ")
		for to, tensor := range feeds {
			glog.V(3).Infof("\t%s: %v", to.Op.Name(), tensor.Shape())
		}
	}

	sess := s.NextSession()
	results, err := sess.Run(feeds, fetches, nil)
	if err != nil {
		log.Panicf("Prediction failed: %v", err)
	}

	// Copy over resulting scores.
	scores := results[0].Value().([]float32)
	if len(scores) != ab.Len() {
		log.Panicf("Expected %d scores (=number of boards given), got %d",
			ab.Len(), len(scores))
	}
	for ii, score := range scores {
		ab.requests[ii].score = score
	}

	// Copy over resulting action probabilities
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
	glog.V(1).Infof("Started AutoBatch dispatcher for [%s].", s)
	var ab *AutoBatch
	for req := range s.autoBatchChan {
		if req != onBatchSizeUpdate {
			if ab == nil {
				ab = s.newAutoBatch()
			}
			ab.Append(req)
			glog.V(3).Info("Received scoring request.")
		} else {
			glog.V(1).Infof("[%s] batch size changed to %d", s, s.autoBatchSize)
		}
		if ab != nil && ab.Len() >= s.autoBatchSize {
			go s.autoBatchScoreAndDeliver(ab)
			ab = nil
		}
	}
}
