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
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
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

type Scorer struct {
	Basename string
	graph    *tf.Graph
	sess     *tf.Session
	mu       sync.Mutex

	BoardFeatures, BoardLabels    tf.Output
	BoardPredictions, BoardLosses tf.Output

	ActionsBoardIndices, ActionsFeatures             tf.Output
	ActionsSourceCenter, ActionsSourceNeighbourhood  tf.Output
	ActionsTargetCenter, ActionsTargetNeighbourhood  tf.Output
	ActionsPredictions, ActionsLosses, ActionsLabels tf.Output

	LearningRate, CheckpointFile, TotalLoss tf.Output
	InitOp, TrainOp, SaveOp, RestoreOp      *tf.Operation

	version int // Uses the number of input features used.
}

// Data used for parsing of player options.
type ParsingData struct {
	UseTensorFlow, ForceCPU bool
}

func NewParsingData() (data interface{}) {
	return &ParsingData{}
}

func FinalizeParsing(data interface{}, player *players.SearcherScorerPlayer) {
	d := data.(*ParsingData)
	if d.UseTensorFlow {
		player.Learner = New(player.ModelFile, d.ForceCPU)
		player.Scorer = player.Learner
	}
}

func ParseParam(data interface{}, key, value string) {
	d := data.(*ParsingData)
	if key == "tf" {
		d.UseTensorFlow = true
	} else if key == "tf_cpu" {
		d.ForceCPU = true
	} else {
		log.Panicf("Unknown parameter '%s=%s' passed to tensorflow module.", key, value)
	}
}

func init() {
	players.RegisterPlayerParameter("tf", "tf", NewParsingData, ParseParam, FinalizeParsing)
	players.RegisterPlayerParameter("tf", "tf_cpu", NewParsingData, ParseParam, FinalizeParsing)
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
func New(basename string, forceCPU bool) *Scorer {
	// Load graph definition (as bytes) and import into current graph.
	graphDefFilename := fmt.Sprintf("%s.pb", basename)
	graphDef, err := ioutil.ReadFile(graphDefFilename)
	if err != nil {
		log.Panicf("Failed to read %q: %v", graphDefFilename, err)
	}

	// Create the one graph and session we will use all time.
	graph := tf.NewGraph()
	sessionOptions := &tf.SessionOptions{}
	if forceCPU || CpuOnly {
		// TODO this doesn't work .... :(
		// Instead use:
		//    export CUDA_VISIBLE_DEVICES=-1
		// Before starting the program.
		var config tfconfig.ConfigProto
		config.DeviceCount = map[string]int32{"GPU": 0}
		data, err := proto.Marshal(&config)
		if err != nil {
			log.Panicf("Failed to serialize tf.ConfigProto: %v", err)
		}
		sessionOptions.Config = data
	}
	sess, err := tf.NewSession(graph, sessionOptions)
	if err != nil {
		log.Panicf("Failed to create tensorflow session: %v", err)
	}
	devices, _ := sess.ListDevices()
	glog.Infof("List of available devices: %v", devices)

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
	op := func(tensorName string) *tf.Operation {
		return graph.Operation(tensorName)
	}

	s := &Scorer{
		Basename: absBasename,
		graph:    graph,
		sess:     sess,

		// Board tensors.
		BoardFeatures:    t0("board_features"),
		BoardLabels:      t0("board_labels"),
		BoardPredictions: t0("board_predictions"),
		BoardLosses:      t0("board_losses"),

		// Actions related tensors.
		ActionsBoardIndices:        t0("actions_board_indices"),
		ActionsFeatures:            t0("actions_features"),
		ActionsSourceCenter:        t0("actions_source_center"),
		ActionsSourceNeighbourhood: t0("actions_source_neighbourhood"),
		ActionsTargetCenter:        t0("actions_target_center"),
		ActionsTargetNeighbourhood: t0("actions_target_neighbourhood"),
		ActionsPredictions:         t0("actions_predictions"),
		ActionsLosses:              t0("actions_losses"),
		ActionsLabels:              t0("actions_labels"),

		// Global parameters.
		LearningRate:   t0("learning_rate"),
		CheckpointFile: t0("save/Const"),
		TotalLoss:      t0("mean_loss"),

		// Ops.
		InitOp:    op("init"),
		TrainOp:   op("train"),
		SaveOp:    op("save/control_dependency"),
		RestoreOp: op("save/restore_all"),
	}
	// Notice there must be a bug in the library that prevents it from taking
	// tf.int32.
	glog.V(2).Infof("ActionsBoardIndices: type=%s", dataType(s.ActionsBoardIndices))

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

	return s
}

func (s *Scorer) String() string {
	return fmt.Sprintf("TensorFlow model in '%s'", s.Basename)
}

func (s *Scorer) CheckpointBase() string {
	return fmt.Sprintf("%s.checkpoint", s.Basename)
}

func (s *Scorer) CheckpointFiles() (string, string) {
	return fmt.Sprintf("%s.checkpoint.index", s.Basename), fmt.Sprintf("%s.checkpoint.data-00000-of-00001", s.Basename)

}

func (s *Scorer) Restore() error {
	t, err := tf.NewTensor(s.CheckpointBase())
	if err != nil {
		log.Panicf("Failed to create tensor: %v", err)
	}
	feeds := map[tf.Output]*tf.Tensor{
		s.CheckpointFile: t,
	}
	_, err = s.sess.Run(feeds, nil, []*tf.Operation{s.RestoreOp})
	return err
}

func (s *Scorer) Init() error {
	_, err := s.sess.Run(nil, nil, []*tf.Operation{s.InitOp})
	return err
}

func (s *Scorer) Version() int {
	return s.version
}

func (s *Scorer) Score(b *Board) (score float32, actionProbs []float32) {
	boards := []*Board{b}
	scores, actionProbsBatch := s.BatchScore(boards)
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

func (s *Scorer) buildFeeds(boards []*Board) (feeds map[tf.Output]*tf.Tensor, totalNumActions int) {
	totalNumActions = 0
	for _, board := range boards {
		totalNumActions += len(board.Derived.Actions)
	}
	glog.V(2).Infof("BatchScore: #boards, #actions=[%d, %d]", len(boards), totalNumActions)

	// Initialize Go objects, that need to be copied to tensors.
	boardFeatures := make([][]float32, len(boards))
	actionsBoardIndices := make([]int64, 0, totalNumActions) // Go tensorflow implementation is broken for int32.
	actionsFeatures := make([][1]float32, 0, totalNumActions)
	actionsSourceCenter := make([][]float32, 0, totalNumActions)
	actionsSourceNeighbourhood := make([][6][]float32, 0, totalNumActions)
	actionsTargetCenter := make([][]float32, 0, totalNumActions)
	actionsTargetNeighbourhood := make([][6][]float32, 0, totalNumActions)

	// Generate features in Go slices.
	for boardIdx, board := range boards {
		boardFeatures[boardIdx] = ai.FeatureVector(board, s.version)
		for _, action := range board.Derived.Actions {
			af := ai.NewActionFeatures(board, action, s.version)
			actionsBoardIndices = append(actionsBoardIndices, int64(boardIdx))
			actionsFeatures = append(actionsFeatures, [1]float32{af.Move})
			actionsSourceCenter = append(actionsSourceCenter, af.SourceFeatures.Center)
			actionsSourceNeighbourhood = append(actionsSourceNeighbourhood,
				af.SourceFeatures.Sections)
			actionsTargetCenter = append(actionsTargetCenter, af.TargetFeatures.Center)
			actionsTargetNeighbourhood = append(actionsTargetNeighbourhood,
				af.TargetFeatures.Sections)

		}
	}

	// Convert Go slices to tensors.
	feeds = map[tf.Output]*tf.Tensor{
		s.BoardFeatures:              mustTensor(boardFeatures),
		s.ActionsBoardIndices:        mustTensor(actionsBoardIndices),
		s.ActionsFeatures:            mustTensor(actionsFeatures),
		s.ActionsSourceCenter:        mustTensor(actionsSourceCenter),
		s.ActionsSourceNeighbourhood: mustTensor(actionsSourceNeighbourhood),
		s.ActionsTargetCenter:        mustTensor(actionsTargetCenter),
		s.ActionsTargetNeighbourhood: mustTensor(actionsTargetNeighbourhood),
	}
	return
}

func (s *Scorer) BatchScore(boards []*Board) (scores []float32, actionProbsBatch [][]float32) {
	// Build feeds to TF model.
	feeds, totalNumActions := s.buildFeeds(boards)
	fetches := []tf.Output{s.BoardPredictions, s.ActionsPredictions}

	// Evaluate: at most one evaluation at a same time.
	s.mu.Lock()
	results, err := s.sess.Run(feeds, fetches, nil)
	s.mu.Unlock()
	if err != nil {
		log.Panicf("Prediction failed: %v", err)
	}

	// Copy over resulting tensors.
	scores = results[0].Value().([]float32)
	actionProbsBatch = make([][]float32, len(boards))
	allActionsProbs := results[1].Value().([]float32)
	if len(scores) != len(boards) {
		log.Panicf("Expected %d scores (=number of boards given), got %d",
			len(boards), len(scores))
	}
	if len(allActionsProbs) != totalNumActions {
		log.Panicf("Expected %d actions (from %d boards), got %d",
			totalNumActions, len(boards), len(allActionsProbs))
	}
	for boardIdx, board := range boards {
		actionProbsBatch[boardIdx] = allActionsProbs[:len(board.Derived.Actions)]
		allActionsProbs = allActionsProbs[len(board.Derived.Actions):]
	}
	return
}

func (s *Scorer) Learn(boards []*Board, boardLabels []float32, actionsLabels []int, learningRate float32, steps int) (loss float32) {
	feeds, totalNumActions := s.buildFeeds(boards)

	// Feed also the labels.
	actionsOneHotLabels := make([]float32, totalNumActions)
	actionsIdx := 0
	for boardIdx, board := range boards {
		actionsOneHotLabels[actionsIdx+actionsLabels[boardIdx]] = 1.0
		actionsIdx += len(board.Derived.Actions)
	}
	if actionsIdx != totalNumActions {
		log.Panicf("Expected %d actions in total, got %d", totalNumActions, actionsIdx)
	}
	feeds[s.BoardLabels] = mustTensor(boardLabels)
	feeds[s.ActionsPredictions] = mustTensor(actionsOneHotLabels)
	feeds[s.LearningRate] = mustTensor(learningRate)

	// Loop over steps.
	s.mu.Lock()
	defer s.mu.Unlock()
	for step := 0; step < steps; step++ {
		if _, err := s.sess.Run(feeds, nil, []*tf.Operation{s.TrainOp}); err != nil {
			log.Panicf("TensorFlow trainOp failed: %v", err)
		}
	}

	fetches := []tf.Output{s.TotalLoss}
	results, err := s.sess.Run(feeds, fetches, nil)
	if err != nil {
		log.Panicf("Loss evaluation failed: %v", err)
	}
	return results[0].Value().(float32)
}

func (s *Scorer) Save() {
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
	if _, err := s.sess.Run(feeds, nil, []*tf.Operation{s.SaveOp}); err != nil {
		log.Panicf("Failed to checkpoint (save) file to %s: %v", s.CheckpointBase(), err)
	}
}
