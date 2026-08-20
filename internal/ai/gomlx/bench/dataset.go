package bench

import (
	"bytes"
	_ "embed"
	"encoding/gob"
	"os"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/pkg/errors"
)

//go:embed match_features.bin
var EmbeddedMatchFeatures []byte

// LoadEmbedded loads the dataset embedded in the binary.
func LoadEmbedded() (*MatchDataset, error) {
	return Load(EmbeddedMatchFeatures)
}

// RawTensor stores flat data and shape info to reconstruct tensors without GoMLX AST or Board dependency.
type RawTensor struct {
	DType   dtypes.DType
	Shape   []int
	Float32 []float32
	Int32   []int32
}

// ToTensor converts RawTensor back to a *tensors.Tensor.
func (rt *RawTensor) ToTensor() *tensors.Tensor {
	sh := shapes.Make(rt.DType, rt.Shape...)
	t := tensors.FromShape(sh)
	if rt.DType == dtypes.Float32 && len(rt.Float32) > 0 {
		tensors.MutableFlatData(t, func(flat []float32) {
			copy(flat, rt.Float32)
		})
	} else if rt.DType == dtypes.Int32 && len(rt.Int32) > 0 {
		tensors.MutableFlatData(t, func(flat []int32) {
			copy(flat, rt.Int32)
		})
	}
	return t
}

// FromTensor converts a *tensors.Tensor to a RawTensor.
func FromTensor(t *tensors.Tensor) RawTensor {
	sh := t.Shape()
	rt := RawTensor{
		DType: sh.DType,
		Shape: append([]int(nil), sh.Dimensions...),
	}
	if sh.DType == dtypes.Float32 {
		rt.Float32 = tensors.MustCopyFlatData[float32](t)
	} else if sh.DType == dtypes.Int32 {
		rt.Int32 = tensors.MustCopyFlatData[int32](t)
	}
	return rt
}

// StepFeatures contains the captured inputs for FNN and A0FNN models at a given step.
type StepFeatures struct {
	FNNInputs   []RawTensor // FNN board scorer inputs: [boardFeatures, numBoards]
	A0ValInputs []RawTensor // A0FNN value inputs: [boardsFeatures, numBoards]
	A0PolInputs []RawTensor // A0FNN policy inputs: [boardFeatures, numBoards, actionsFeatures, actionsToBoardIdx, numActions]
}

// MatchDataset holds a sequence of step features recorded during an entire match.
type MatchDataset struct {
	Steps []StepFeatures
}

// Save writes the dataset to a file using gob encoding.
func (ds *MatchDataset) Save(path string) error {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(ds); err != nil {
		return errors.Wrap(err, "failed to encode MatchDataset")
	}
	return os.WriteFile(path, buf.Bytes(), 0644)
}

// Load loads the dataset from file bytes.
func Load(data []byte) (*MatchDataset, error) {
	var ds MatchDataset
	dec := gob.NewDecoder(bytes.NewReader(data))
	if err := dec.Decode(&ds); err != nil {
		return nil, errors.Wrap(err, "failed to decode MatchDataset")
	}
	return &ds, nil
}
