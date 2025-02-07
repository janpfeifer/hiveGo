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
	"bufio"
	"flag"
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/features"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	tfconfig "github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	// Set this to true to force to use CPU, even when GPU is avaialble.
	CpuOnly = false

	// These are default values for tensorflow models' tensors (if they are
	// set as placeholders). They can be overwritten using the flag --tf_params.
	DefaultModelParams = map[string]float32{
		"learning_rate":              1e-5,
		"actions_loss_ratio":         0.005,
		"l2_regularization":          1e-5,
		"self_supervision":           0.0,
		"td_lambda":                  1.0,
		"calibration_regularization": 0.0,
		"dropout_keep_probability":   1.0,
	}
)

const (
	INTER_OP_PARALLELISM = 4
	INTRA_OP_PARALLELISM = 4
)

var (
	Flag_learnBatchSize = flag.Int("tf_batch_size", 0,
		"Batch size when learning: this is the number of boards, not actions. There is usually 100/1 ratio of "+
			"actions per board. Examples are shuffled before being batched. 0 means no batching.")
	flag_tfParams = flag.String("tf_params", "",
		"Comma separated list of `key=value` pairs where `key` is a string (name of a tensor), and "+
			"`value` is a float32 value. These are assumed to be model's placeholders "+
			"that will be set with the given value on all inference/train session runs. If "+
			"tensor named `key` does not exist in model, it is reported but ignored.")
	flag_tfParamsFile = flag.String("tf_params_file", "",
		"Similar to tf_params, but instead read `key=values` pairs from given file. "+
			"The file is re-read before each call to learn(), so parameters can be changed "+
			"dynamically during long training periods.")
	flag_tfGpuMemoryFraction = flag.Float64("tf_gpu_mem", 0.3,
		"Fraction of the available GPU memory to use.")
)

type tfOutputTensor struct {
	key   tf.Output
	value *tf.Tensor
}

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
	BoardMovesToEnd               tf.Output
	FullBoard                     tf.Output
	BoardPredictions, BoardLosses tf.Output

	ActionsBoardIndices, ActionsIsMove       tf.Output
	ActionsSrcPositions, ActionsTgtPositions tf.Output
	ActionsPieces, ActionsLabels             tf.Output
	ActionsPredictions, ActionsLosses        tf.Output

	IsTraining, CheckpointFile         tf.Output
	GlobalStep, TotalLoss              tf.Output
	InitOp, TrainOp, SaveOp, RestoreOp *tf.Operation

	// Params set by flag --tf_params.
	Params                     map[string]*tfOutputTensor
	paramsFileLastModifiedTime time.Time

	version int
	// Uses the number of input features used.
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
//
// sessionPoolSize defines the size for a pool of Sessions to use.
// forceCPU for forcing all the TensorFlow ops into the CPU.
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
		BoardMovesToEnd:  t0opt("board_moves_to_end"),
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
		IsTraining:     t0opt("is_training"),
		CheckpointFile: t0("save/Const"),
		TotalLoss:      t0("mean_loss"),
		GlobalStep:     t0("global_step"),

		// Ops.
		InitOp:    op("init"),
		TrainOp:   op("train"),
		SaveOp:    op("save/control_dependency"),
		RestoreOp: op("save/restore_all"),

		Params: make(map[string]*tfOutputTensor),
	}

	// Model must have all actions tensors to be considered actions classifier.
	s.isActionsClassifier =
		s.ActionsBoardIndices.Op != nil && s.ActionsIsMove.Op != nil &&
			s.ActionsSrcPositions.Op != nil && s.ActionsTgtPositions.Op != nil &&
			s.ActionsPieces.Op != nil && s.ActionsLabels.Op != nil &&
			s.ActionsPredictions.Op != nil && s.ActionsLosses.Op != nil
	if !s.isActionsClassifier {
		klog.Infof("%s can not be used for actions classification.", s)
	}

	// Set version to the size of the input.
	s.version = int(s.BoardFeatures.Shape().Size(1))
	klog.V(1).Infof("TensorFlow model's version=%d", s.version)

	// Set generic parameters.
	s.parseFlagParams()
	s.parseFlagParamsFile()

	// Either restore or initialize the network.
	cpIndex, _ := s.CheckpointFiles()
	if _, err := os.Stat(cpIndex); err == nil {
		klog.Infof("Loading model from %s", s.CheckpointBase())
		err = s.Restore()
		if err != nil {
			log.Panicf("Failed to load checkpoint from file %s: %v", s.CheckpointBase(), err)
		}
	} else if os.IsNotExist(err) {
		klog.Infof("Initializing model randomly, since %s not found", s.CheckpointBase())
		err = s.Init()
		if err != nil {
			log.Panicf("Failed to initialize model: %v", err)
		}
	} else {
		log.Panicf("Cannot checkpoint file %s: %v", s.CheckpointBase(), err)
	}

	go s.autoBatchDispatcher()

	klog.Infof("global_step=%d", s.ReadGlobalStep())
	return s
}

func (s *Scorer) setParam(key string, value float32) {
	t0opt := func(tensorName string) (to tf.Output) {
		op := s.graph.Operation(tensorName)
		if op == nil {
			return
		}
		return op.Output(0)
	}

	tfOut := t0opt(key)
	if tfOut.Op != nil {
		if pair, found := s.Params[key]; found {
			klog.Infof("Tensor '%s' updated to %g", key, value)
			pair.value = mustTensor(value)
		} else {
			klog.Infof("Tensor '%s' set to %g", key, value)
			s.Params[key] = &tfOutputTensor{tfOut, mustTensor(value)}
		}
	} else {
		klog.Errorf("Tensor '%s' not found and cannot be set", key)
	}
}

func (s *Scorer) parseFlagParams() {
	keyValues := make(map[string]float32)
	for key, value := range DefaultModelParams {
		keyValues[key] = value
	}
	for _, pair := range strings.Split(*flag_tfParams, ",") {
		if pair == "" {
			continue
		}
		keyValue := strings.Split(pair, "=")
		if len(keyValue) != 2 {
			log.Panicf("Malformed --tf_params entry: [ %s ]", keyValue)
		}
		if v, err := strconv.ParseFloat(keyValue[1], 32); err == nil {
			keyValues[keyValue[0]] = float32(v)
		} else {
			log.Panicf("Malformed --tf_params entry [ %s ]: %v", keyValue, err)
		}
	}
	for key, value := range keyValues {
		s.setParam(key, value)
	}
	return
}

func (s *Scorer) parseFlagParamsFile() {
	// Skip if no file configured.
	if *flag_tfParamsFile == "" {
		return
	}

	// Check if file changed since last time it was parsed.
	info, err := os.Stat(*flag_tfParamsFile)
	if err != nil {
		log.Panicf("Cannot stat file given by --tf_params_file=%s: %v", *flag_tfParamsFile, err)
	}
	if info.ModTime().Equal(s.paramsFileLastModifiedTime) {
		// File not modified.
		return
	}
	s.paramsFileLastModifiedTime = info.ModTime()
	klog.Infof("Parsing TensorFlow parameters in %s for model %s", *flag_tfParamsFile, s)

	// Parse key value pairs.
	keyValues := make(map[string]float32)
	file, err := os.Open(*flag_tfParamsFile)
	if err != nil {
		log.Panicf("Failed reading file given by --tf_params_file=%s: %v", *flag_tfParamsFile, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		pair := scanner.Text()
		if pair == "" {
			continue
		}
		keyValue := strings.Split(pair, "=")
		if len(keyValue) != 2 {
			log.Panicf("Malformed --tf_params_file entry: [ %s ]", keyValue)
		}
		if v, err := strconv.ParseFloat(keyValue[1], 32); err == nil {
			keyValues[keyValue[0]] = float32(v)
		} else {
			log.Panicf("Malformed --tf_params_file entry [ %s ]: %v", keyValue, err)
		}
	}
	if err := scanner.Err(); err != nil {
		log.Panicf("Failed reading file given by --tf_params_file=%s: %v", *flag_tfParamsFile, err)
	}
	for key, value := range keyValues {
		s.setParam(key, value)
	}
	return
}

func (s *Scorer) IsActionsClassifier() bool {
	return s.isActionsClassifier
}

func (s *Scorer) HasFullBoard() bool {
	return s.FullBoard.Op != nil
}

func (s *Scorer) ReadGlobalStep() int64 {
	sess := s.NextSession()
	res, err := sess.Run(nil, []tf.Output{s.GlobalStep}, nil)
	if err != nil {
		log.Panicf("Can't read GlobalStep: %v", err)
	}
	return res[0].Value().(int64)
}

func createSessionPool(graph *tf.Graph, size int, forceCPU bool) (sessions []*tf.Session) {
	gpuMemFractionLeft := *flag_tfGpuMemoryFraction
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
			klog.Infof("List of available devices: %v", devices)
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
	return fmt.Sprintf("%s.checkpoint.index", s.Basename),
		fmt.Sprintf("%s.checkpoint.data-00000-of-00001", s.Basename)

}

func (s *Scorer) CheckpointFilesForStep(step int64) (string, string) {
	dir := path.Dir(s.Basename)

	return fmt.Sprintf("%s/checkpoints/%09d.index", dir, step),
		fmt.Sprintf("%s/checkpoints/%09d.data-00000-of-00001", dir, step)
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
	boardMovesToEnd   []float32
	fullBoardFeatures [][][][]float32 // [batch, height, width, depth]
	boardLabels       []float32

	// Labels
	actionsBoardIndices []int64 // Go tensorflow implementation is broken for int32.
	actionsIsMove       []bool
	actionsSrcPositions [][2]int64
	actionsTgtPositions [][2]int64
	actionsPieces       [][NumPieceTypes]float32
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
		batchWidth, batchHeight = features.SuggestedFullBoardWidth, features.SuggestedFullBoardHeight
		for _, b := range boards {
			w, h := features.FullBoardDimensions(b)
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
			if board.NumActions() > 1 {
				// We ignore when there are no valid actions, or if there is
				// only one obvious action.
				fc.totalNumActions += board.NumActions()
			}
		}
		fc.actionsBoardIndices = make([]int64, fc.totalNumActions) // Go tensorflow implementation is broken for int32.
		fc.actionsIsMove = make([]bool, fc.totalNumActions)
		fc.actionsSrcPositions = make([][2]int64, fc.totalNumActions)
		fc.actionsTgtPositions = make([][2]int64, fc.totalNumActions)
		fc.actionsPieces = make([][NumPieceTypes]float32, fc.totalNumActions)
	}

	// Generate features in Go slices.
	actionIdx := 0
	for boardIdx, board := range boards {
		fc.boardFeatures[boardIdx] = features.FeatureVector(board, s.version)
		if s.HasFullBoard() {
			fc.fullBoardFeatures[boardIdx] = features.MakeFullBoardFeatures(board, batchWidth, batchHeight)
		}
		if scoreActions && board.NumActions() > 1 {
			for _, action := range board.Derived.Actions {
				fc.actionsBoardIndices[actionIdx] = int64(boardIdx)
				fc.actionsIsMove[actionIdx] = action.Move
				fc.actionsSrcPositions[actionIdx] = features.PosToFullBoardPosition(board, action.SourcePos)
				fc.actionsTgtPositions[actionIdx] = features.PosToFullBoardPosition(board, action.TargetPos)
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
		if klog.V(2) {
			f := fc.fullBoardFeatures
			shape := []int{len(f), len(f[0]), len(f[0][0]), len(f[0][0][0])}
			klog.Infof("fullBoardFeatures shape: %v", shape)
			if scoreActions {
				klog.Infof("ActionsBoardIndices shape: %v", []int{len(fc.actionsBoardIndices)})
				klog.Infof("ActionsIsMove shape: %v", []int{len(fc.actionsIsMove)})
				klog.Infof("ActionsActionsPieces shape: %v", []int{len(fc.actionsPieces), len(fc.actionsPieces[0])})
			}
		}
		feeds[s.FullBoard] = mustTensor(fc.fullBoardFeatures)
	}
	for _, pair := range s.Params {
		feeds[pair.key] = pair.value
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
	if klog.V(3) {
		klog.V(3).Infof("Feeded tensors: ")
		for to, tensor := range feeds {
			klog.V(3).Infof("\t%s: %v = %v", to.Op.Name(), tensor.Shape(),
				tensor.Value())
		}
	}

	sess := s.NextSession()
	results, err := sess.Run(feeds, fetches, nil)
	if err != nil {
		log.Panicf("Prediction failed: %v", err)
	}

	// Clone over resulting tensors.
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

func (s *Scorer) Save() {
	globalStep := s.ReadGlobalStep()
	klog.Infof("Saving %s, checkpointing at global_step=%09d", s, globalStep)
	if len(s.sessionPool) > 1 {
		log.Panicf("SessionPool doesn't support saving. You probably should use sessionPoolSize=1 in this case.")
	}

	// Backup previous checkpoint.
	index, data := s.CheckpointFiles()
	if _, err := os.Stat(index); err == nil {
		if err := os.Rename(index, index+"~"); err != nil {
			klog.Errorf("Failed to backup %s to %s~: %v", index, index, err)
		}
		if err := os.Rename(data, data+"~"); err != nil {
			klog.Errorf("Failed to backup %s to %s~: %v", data, data, err)
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

	// Link files to version with global step.
	index2, data2 := s.CheckpointFilesForStep(globalStep)
	linked := false
	if _, err1 := os.Stat(data2); os.IsNotExist(err1) {
		if _, err2 := os.Stat(index2); os.IsNotExist(err2) {
			if err := os.Link(data, data2); err != nil {
				log.Panicf("Failed to link %s to %s: %v", data, data2, err)
			}
			if err := os.Link(index, index2); err != nil {
				log.Panicf("Failed to link %s to %s: %v", index, index2, err)
			}
			linked = true
		}
	}
	if !linked {
		klog.Errorf("Failed to link saved model in %s to global_step=%09d", s, globalStep)
	}
}
