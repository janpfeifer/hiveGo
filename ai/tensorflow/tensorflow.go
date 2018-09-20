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
//          ${REPOS}/tensorflow/tensorflow/core/protobuf/config.proto
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

	input, label, learningRate, checkpointFile tf.Output
	output, loss                               tf.Output
	initOp, trainOp, saveOp, restoreOp         *tf.Operation
}

// New creates a new Scorer by reading model's graph `basename`.pb,
// and checkpoints from `basename`.checkpoint
func New(basename string, cpu bool) *Scorer {
	// Load graph definition (as bytes) and import into current graph.
	graphDefFilename := fmt.Sprintf("%s.pb", basename)
	graphDef, err := ioutil.ReadFile(graphDefFilename)
	if err != nil {
		log.Panicf("Failed to read %q: %v", graphDefFilename, err)
	}

	// Create the one graph and session we will use all time.
	graph := tf.NewGraph()
	sessionOptions := &tf.SessionOptions{}
	if CpuOnly {
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

	s := &Scorer{
		Basename: absBasename,
		graph:    graph,
		sess:     sess,

		// Input tensors, placeholders.
		input:          graph.Operation("input").Output(0),
		label:          graph.Operation("label").Output(0),
		learningRate:   graph.Operation("learning_rate").Output(0),
		checkpointFile: graph.Operation("save/Const").Output(0),

		// Output tensors.
		output: graph.Operation("output").Output(0),
		loss:   graph.Operation("loss").Output(0),

		// Ops
		initOp:    graph.Operation("init"),
		trainOp:   graph.Operation("train"),
		saveOp:    graph.Operation("save/control_dependency"),
		restoreOp: graph.Operation("save/restore_all"),
	}

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
		s.Init()
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
		s.checkpointFile: t,
	}
	_, err = s.sess.Run(feeds, nil, []*tf.Operation{s.restoreOp})
	return err
}

func (s *Scorer) Init() error {
	_, err := s.sess.Run(nil, nil, []*tf.Operation{s.initOp})
	return err
}

func (s *Scorer) UnlimitedBatchScore(batch [][]float32) []float32 {
	glog.V(2).Infof("UnlimitedBatchScore: batch.size=[%d, %d]", len(batch), len(batch[0]))
	batchTensor, err := tf.NewTensor(batch)
	if err != nil {
		log.Panicf("Failed to create tensor: %v", err)
	}
	feeds := map[tf.Output]*tf.Tensor{s.input: batchTensor}
	fetches := []tf.Output{s.output}
	s.mu.Lock()
	defer s.mu.Unlock()
	results, err := s.sess.Run(feeds, fetches, nil)
	if err != nil {
		log.Panicf("Prediction failed: %v", err)
	}
	return results[0].Value().([]float32)
}

func (s *Scorer) Score(b *Board) float32 {
	features := [][]float32{ai.FeatureVector(b)}
	scores := s.UnlimitedBatchScore(features)
	score := scores[0]
	// if score > 9.8 {
	// 	score = 9.8
	// } else if score < -9.8 {
	// 	score = -9.8
	// }
	return score
}

func (s *Scorer) BatchScore(boards []*Board) []float32 {
	features := make([][]float32, len(boards))
	for ii, board := range boards {
		features[ii] = ai.FeatureVector(board)
	}
	scores := s.UnlimitedBatchScore(features)
	for ii, score := range scores {
		if score > 9.8 {
			scores[ii] = 9.8
		} else if score < -9.8 {
			scores[ii] = -9.8
		}
	}
	return scores
}

func (s *Scorer) Learn(learningRate float32, examples []ai.LabeledExample, steps int) float32 {
	glog.V(1).Infof("Learn: batch.size=[%d, %d]", len(examples), len(examples[0].Features))

	// Extract features and labels.
	features := make([][]float32, len(examples))
	labels := make([]float32, len(examples))
	for ii, example := range examples {
		features[ii] = example.Features
		labels[ii] = example.Label
	}

	// Prepare input feeds.
	inputTensor, err := tf.NewTensor(features)
	if err != nil {
		log.Panicf("Failed to create tensor: %v", err)
	}
	labelTensor, err := tf.NewTensor(labels)
	if err != nil {
		log.Panicf("Failed to create tensor: %v", err)
	}
	learningRateTensor, err := tf.NewTensor(learningRate)
	if err != nil {
		log.Panicf("Failed to create tensor: %v", err)
	}
	feeds := map[tf.Output]*tf.Tensor{
		s.input:        inputTensor,
		s.label:        labelTensor,
		s.learningRate: learningRateTensor,
	}

	// Loop over steps.
	s.mu.Lock()
	defer s.mu.Unlock()
	for step := 0; step < steps; step++ {
		if _, err = s.sess.Run(feeds, nil, []*tf.Operation{s.trainOp}); err != nil {
			log.Panicf("TensorFlow trainOp failed: %v", err)
		}
	}

	fetches := []tf.Output{s.loss}
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
	feeds := map[tf.Output]*tf.Tensor{s.checkpointFile: t}
	if _, err := s.sess.Run(feeds, nil, []*tf.Operation{s.saveOp}); err != nil {
		log.Panicf("Failed to checkpoint (save) file to %s: %v", s.CheckpointBase(), err)
	}
}
