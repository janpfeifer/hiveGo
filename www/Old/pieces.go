package Old

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/internal/state"
	Old2 "github.com/janpfeifer/hiveGo/www/Old"

	"github.com/gopherjs/gopherjs/js"
	"github.com/gopherjs/jquery"
)

var (
	// Patterns for pieces.
	boardPiecesPatterns    []*js.Object
	boardPiecesImages      []*js.Object
	OffBoardPiecesPatterns []*js.Object
	OffBoardPiecesImages   []*js.Object

	// Map of all pieces currently on board.
	piecesOnBoard      = make(map[state.Pos][]*PieceOnScreen)
	piecesOnBoardIndex = 0
)

const (
	// Interface visual constants:

	PieceDrawingScale = 1.464
	HexStrokeWidth    = 2.0
	ImageBaseSize     = 0.0488
)

type PieceOnScreen struct {
	Index     int
	Player    state.PlayerNum
	StackPos  int
	Piece     state.PieceType
	Hex, Rect jquery.JQuery
}

func (g *Old2.Game) Place(player state.PlayerNum, action state.Action) {
	pos := action.TargetPos
	stack := piecesOnBoard[pos]
	pons := &PieceOnScreen{
		Index:  piecesOnBoardIndex,
		Player: player,
		Piece:  action.Piece,
		Hex: jq(Old2.CreateSVG("polygon", Old2.Attrs{
			"stroke":       "black",
			"stroke-width": HexStrokeWidth * ui.Scale,
			"fill":         PlayerBackgroundColor(player),
			"fill-opacity": 1.0,
		})),
		Rect: jq(Old2.CreateSVG("rect", Old2.Attrs{
			"stroke":         "black",
			"stroke-width":   0,
			"border":         0,
			"padding":        0,
			"fill":           fmt.Sprintf("url(#%s)", PieceToPatternId(BOARD, action.Piece)),
			"pointer-events": "none",
		})),
	}
	pons.MoveTo(pos, len(stack))
	piecesOnBoard[pos] = append(stack, pons)
	stackPos := len(stack)
	piecesOnBoardIndex++

	// Make sure the new piece is under other pieces that are higher.
	var ponsAbove *PieceOnScreen
	for _, pieces := range piecesOnBoard {
		if len(pieces) > stackPos+1 {
			for _, tmpPons := range pieces[stackPos+1:] {
				if ponsAbove == nil || tmpPons.Index < ponsAbove.Index {
					ponsAbove = tmpPons
				}
			}
		}
	}
	if ponsAbove == nil {
		fmt.Printf("Appending piece: %v\n", pons.Hex)
		Old2.BoardGroup.Append(pons.Hex)
		Old2.BoardGroup.Append(pons.Rect)
	} else {
		fmt.Printf("Inserting piece before another: %v\n", ponsAbove)
		pons.Hex.InsertBefore(ponsAbove.Hex)
		pons.Rect.InsertBefore(ponsAbove.Hex)
	}

	// Connect click to selection.
	pons.Hex.On(jquery.MOUSEUP, func(e jquery.Event) {
		g.OnSelectOnBoard(pons, pos)
	})
}
