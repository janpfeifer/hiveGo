// Code generated by "enumer -type=BoardId -trimprefix=Id -values -text -json -yaml features.go"; DO NOT EDIT.

package features

import (
	"encoding/json"
	"fmt"
	"strings"
)

const _BoardIdName = "NumOffboardOpponentNumOffboardNumSurroundingQueenOpponentNumSurroundingQueenNumCanMoveOpponentNumCanMoveNumThreateningMovesOpponentNumThreateningMovesMovesToDrawNumSingleQueenCoveredAverageDistanceToQueenOpponentAverageDistanceToQueenNumPlacementPositionsMoveNumberNumFeatureIds"

var _BoardIdIndex = [...]uint16{0, 11, 30, 49, 76, 86, 104, 123, 150, 161, 170, 182, 204, 234, 255, 265, 278}

const _BoardIdLowerName = "numoffboardopponentnumoffboardnumsurroundingqueenopponentnumsurroundingqueennumcanmoveopponentnumcanmovenumthreateningmovesopponentnumthreateningmovesmovestodrawnumsinglequeencoveredaveragedistancetoqueenopponentaveragedistancetoqueennumplacementpositionsmovenumbernumfeatureids"

func (i BoardId) String() string {
	if i >= BoardId(len(_BoardIdIndex)-1) {
		return fmt.Sprintf("BoardId(%d)", i)
	}
	return _BoardIdName[_BoardIdIndex[i]:_BoardIdIndex[i+1]]
}

func (BoardId) Values() []string {
	return BoardIdStrings()
}

// An "invalid array index" compiler error signifies that the constant values have changed.
// Re-run the stringer command to generate them again.
func _BoardIdNoOp() {
	var x [1]struct{}
	_ = x[IdNumOffboard-(0)]
	_ = x[IdOpponentNumOffboard-(1)]
	_ = x[IdNumSurroundingQueen-(2)]
	_ = x[IdOpponentNumSurroundingQueen-(3)]
	_ = x[IdNumCanMove-(4)]
	_ = x[IdOpponentNumCanMove-(5)]
	_ = x[IdNumThreateningMoves-(6)]
	_ = x[IdOpponentNumThreateningMoves-(7)]
	_ = x[IdMovesToDraw-(8)]
	_ = x[IdNumSingle-(9)]
	_ = x[IdQueenCovered-(10)]
	_ = x[IdAverageDistanceToQueen-(11)]
	_ = x[IdOpponentAverageDistanceToQueen-(12)]
	_ = x[IdNumPlacementPositions-(13)]
	_ = x[IdMoveNumber-(14)]
	_ = x[IdNumFeatureIds-(15)]
}

var _BoardIdValues = []BoardId{IdNumOffboard, IdOpponentNumOffboard, IdNumSurroundingQueen, IdOpponentNumSurroundingQueen, IdNumCanMove, IdOpponentNumCanMove, IdNumThreateningMoves, IdOpponentNumThreateningMoves, IdMovesToDraw, IdNumSingle, IdQueenCovered, IdAverageDistanceToQueen, IdOpponentAverageDistanceToQueen, IdNumPlacementPositions, IdMoveNumber, IdNumFeatureIds}

var _BoardIdNameToValueMap = map[string]BoardId{
	_BoardIdName[0:11]:         IdNumOffboard,
	_BoardIdLowerName[0:11]:    IdNumOffboard,
	_BoardIdName[11:30]:        IdOpponentNumOffboard,
	_BoardIdLowerName[11:30]:   IdOpponentNumOffboard,
	_BoardIdName[30:49]:        IdNumSurroundingQueen,
	_BoardIdLowerName[30:49]:   IdNumSurroundingQueen,
	_BoardIdName[49:76]:        IdOpponentNumSurroundingQueen,
	_BoardIdLowerName[49:76]:   IdOpponentNumSurroundingQueen,
	_BoardIdName[76:86]:        IdNumCanMove,
	_BoardIdLowerName[76:86]:   IdNumCanMove,
	_BoardIdName[86:104]:       IdOpponentNumCanMove,
	_BoardIdLowerName[86:104]:  IdOpponentNumCanMove,
	_BoardIdName[104:123]:      IdNumThreateningMoves,
	_BoardIdLowerName[104:123]: IdNumThreateningMoves,
	_BoardIdName[123:150]:      IdOpponentNumThreateningMoves,
	_BoardIdLowerName[123:150]: IdOpponentNumThreateningMoves,
	_BoardIdName[150:161]:      IdMovesToDraw,
	_BoardIdLowerName[150:161]: IdMovesToDraw,
	_BoardIdName[161:170]:      IdNumSingle,
	_BoardIdLowerName[161:170]: IdNumSingle,
	_BoardIdName[170:182]:      IdQueenCovered,
	_BoardIdLowerName[170:182]: IdQueenCovered,
	_BoardIdName[182:204]:      IdAverageDistanceToQueen,
	_BoardIdLowerName[182:204]: IdAverageDistanceToQueen,
	_BoardIdName[204:234]:      IdOpponentAverageDistanceToQueen,
	_BoardIdLowerName[204:234]: IdOpponentAverageDistanceToQueen,
	_BoardIdName[234:255]:      IdNumPlacementPositions,
	_BoardIdLowerName[234:255]: IdNumPlacementPositions,
	_BoardIdName[255:265]:      IdMoveNumber,
	_BoardIdLowerName[255:265]: IdMoveNumber,
	_BoardIdName[265:278]:      IdNumFeatureIds,
	_BoardIdLowerName[265:278]: IdNumFeatureIds,
}

var _BoardIdNames = []string{
	_BoardIdName[0:11],
	_BoardIdName[11:30],
	_BoardIdName[30:49],
	_BoardIdName[49:76],
	_BoardIdName[76:86],
	_BoardIdName[86:104],
	_BoardIdName[104:123],
	_BoardIdName[123:150],
	_BoardIdName[150:161],
	_BoardIdName[161:170],
	_BoardIdName[170:182],
	_BoardIdName[182:204],
	_BoardIdName[204:234],
	_BoardIdName[234:255],
	_BoardIdName[255:265],
	_BoardIdName[265:278],
}

// BoardIdString retrieves an enum value from the enum constants string name.
// Throws an error if the param is not part of the enum.
func BoardIdString(s string) (BoardId, error) {
	if val, ok := _BoardIdNameToValueMap[s]; ok {
		return val, nil
	}

	if val, ok := _BoardIdNameToValueMap[strings.ToLower(s)]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to BoardId values", s)
}

// BoardIdValues returns all values of the enum
func BoardIdValues() []BoardId {
	return _BoardIdValues
}

// BoardIdStrings returns a slice of all String values of the enum
func BoardIdStrings() []string {
	strs := make([]string, len(_BoardIdNames))
	copy(strs, _BoardIdNames)
	return strs
}

// IsABoardId returns "true" if the value is listed in the enum definition. "false" otherwise
func (i BoardId) IsABoardId() bool {
	for _, v := range _BoardIdValues {
		if i == v {
			return true
		}
	}
	return false
}

// MarshalJSON implements the json.Marshaler interface for BoardId
func (i BoardId) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

// UnmarshalJSON implements the json.Unmarshaler interface for BoardId
func (i *BoardId) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("BoardId should be a string, got %s", data)
	}

	var err error
	*i, err = BoardIdString(s)
	return err
}

// MarshalText implements the encoding.TextMarshaler interface for BoardId
func (i BoardId) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface for BoardId
func (i *BoardId) UnmarshalText(text []byte) error {
	var err error
	*i, err = BoardIdString(string(text))
	return err
}

// MarshalYAML implements a YAML Marshaler for BoardId
func (i BoardId) MarshalYAML() (interface{}, error) {
	return i.String(), nil
}

// UnmarshalYAML implements a YAML Unmarshaler for BoardId
func (i *BoardId) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}

	var err error
	*i, err = BoardIdString(s)
	return err
}
