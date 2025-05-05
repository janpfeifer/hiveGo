package main

import (
	"fmt"
	"github.com/gomlx/gomlx/types"
	"github.com/janpfeifer/hiveGo/internal/state"
)

// Selections is a component of WebUI that handles the human player selection of an action.
type Selections struct {
	isSelecting, isSelectingSource, isSelectingTarget bool

	// Valid sources of the move (off-board or on-board)
	sourcesOnBoardPons, sourceOffBoardPons []*PieceOnScreen
	selectedSourceIsOffBoard               bool
	selectedSourcePons                     *PieceOnScreen

	//TargetHexes                           []jquery.JQuery
	//TargetActions                         []state.Action

}

// selectionsInit is called when WebUI is constructed.
func (ui *WebUI) selectionsInit() {
}

// AdjustSelections to change in resolution.
func (ui *WebUI) AdjustSelections() {
	strokeWidth := 2.5 * HexStrokeWidth * ui.PixelRatio
	for _, pons := range ui.selections.sourceOffBoardPons {
		SetAttrs(&pons.Hex.Element, Attrs{"stroke-width": strokeWidth})
	}
}

// SelectAction will show all valid actions and wait for a response from the user and return
// only when the user has selected an action.
//
// It uses the current board, which can be set with WebUI.UpdateBoard.
func (ui *WebUI) SelectAction() state.Action {
	ui.selections.isSelecting = true
	ui.selectSource()

	select {}

	ui.selections.isSelecting = false
	return state.Action{}
}

// selectSource displays the valid source positions to move.
func (ui *WebUI) selectSource() {
	board := ui.board
	playerNum := ui.board.NextPlayer

	// Find off-board pieces that are placeable:
	offBoardPieces := types.MakeSet[state.PieceType]()
	for _, action := range board.Derived.Actions {
		if action.Move {
			// We are only interested in the place new piece actions here.
			continue
		}
		offBoardPieces.Insert(action.Piece)
	}
	ui.selections.sourceOffBoardPons = make([]*PieceOnScreen, 0, len(offBoardPieces))
	for pieceType := range offBoardPieces {
		stack := ui.piecesOffBoard[playerNum][pieceType]
		topPons := stack[len(stack)-1]
		ui.selections.sourceOffBoardPons = append(ui.selections.sourceOffBoardPons, topPons)
		ui.makeSelectable(topPons)
	}

	// Reshapes as needed.
	ui.AdjustSelections()
}

// resetSourceSelection will unmake sources as selectable -- because we are moving to selecting the target.
func (ui *WebUI) resetSourceSelection() {
	for _, pons := range ui.selections.sourceOffBoardPons {
		ui.makeUnselectable(pons)
	}
	ui.AdjustOffBoardPieces()
	ui.AdjustOnBoardPieces()
}

// makeSelectable changes the hexagon around the piece to be flashy.
func (ui *WebUI) makeSelectable(pons *PieceOnScreen) {
	SetAttrs(&pons.Hex.Element, Attrs{
		"stroke":           "yellow",
		"stroke-dasharray": "2",
	})
	animate := CreateSVG("animate", Attrs{
		"attributeName": "stroke-dashoffset",
		"from":          0.0,
		"to":            100.0, // Enough to cover the whole hexagon.
		"dur":           "500ms",
		"repeatCount":   "indefinite",
	})
	pons.Hex.AppendChild(&animate.Node)
}

// makeUnselectable changes the hexagon around the piece to be normal.
func (ui *WebUI) makeUnselectable(pons *PieceOnScreen) {
	SetAttrs(&pons.Hex.Element, Attrs{
		"stroke":           "url(#reliefStroke)",
		"stroke-dasharray": nil, // Remove it.
	})

	// Remove animation if there is one.
	animate := pons.Hex.QuerySelector("animate")
	if animate != nil {
		animate.Remove()
	}
}

// OnSelectOffBoardPiece is called when an off-board piece is clicked.
func (ui *WebUI) OnSelectOffBoardPiece(pons *PieceOnScreen) {
	if !ui.selections.isSelecting {
		return
	}
	if ui.selections.isSelectingTarget {
		if pons != ui.selections.selectedSourcePons {
			// Click on a random off-board piece, ignore.
			return
		}
		// Undo source selection.
		// TODO
		return
	}
	var found bool
	for _, srcPons := range ui.selections.sourceOffBoardPons {
		if pons == srcPons {
			found = true
			break
		}
	}
	if !found {
		// Invalid piece: from another player or under the stack.
		return
	}

	// Source piece to place was selected.
	ui.selections.selectedSourceIsOffBoard = true
	ui.selections.selectedSourcePons = pons
	ui.selections.isSelectingSource = false
	ui.selections.isSelectingTarget = true
	fmt.Printf("Selected off-board %s (player #%d)\n", pons.PieceType, pons.Player+1)
	ui.resetSourceSelection()
}
