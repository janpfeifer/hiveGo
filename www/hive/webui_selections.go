package main

import (
	"github.com/gowebapi/webapi/graphics/svg"
	"github.com/gowebapi/webapi/html/htmlevent"
	"github.com/janpfeifer/hiveGo/internal/generics"
	"github.com/janpfeifer/hiveGo/internal/state"
	"k8s.io/klog/v2"
)

// Selections is a component of WebUI that handles the human player selection of an action.
type Selections struct {
	isSelecting, isSelectingSource, isSelectingTarget bool
	selectedAction                                    chan state.Action

	// Valid sources of the move (off-board or on-board)
	sourceOnBoardPons, sourceOffBoardPons []*PieceOnScreen

	// Selected source, valid once isSelectingTarget is true.
	selectedSourceIsOffBoard bool
	selectedSourcePons       *PieceOnScreen
	selectedSourcePos        state.Pos

	// Valid target positions
	targetPons    []*PieceOnScreen
	targetActions []state.Action
}

// selectionsInit is called when WebUI is constructed.
func (ui *WebUI) selectionsInit() {
	ui.selections.selectedAction = make(chan state.Action)
}

// AdjustSelections to change in resolution.
func (ui *WebUI) AdjustSelections() {
	strokeWidth := 2.5 * HexStrokeWidth * ui.PixelRatio
	for _, pons := range ui.selections.sourceOffBoardPons {
		SetAttrs(&pons.Hex.Element, Attrs{"stroke-width": strokeWidth})
	}
	for ii, pons := range ui.selections.targetPons {
		SetAttrs(&pons.Hex.Element, Attrs{"stroke-width": strokeWidth})
		ui.MovePieceTo(pons, ui.selections.targetActions[ii].TargetPos, pons.StackPos)
	}
}

// SelectAction will show all valid actions and wait for a response from the user and return
// only when the user has selected an action.
//
// It uses the current board, which can be set with WebUI.UpdateBoard.
func (ui *WebUI) SelectAction() state.Action {
	ui.selections.isSelecting = true
	ui.selectSource()
	action := <-ui.selections.selectedAction
	ui.selections.isSelecting = false
	return action
}

// selectSource displays the valid source positions to move.
func (ui *WebUI) selectSource() {
	ui.selections.isSelectingSource = true
	ui.selections.isSelectingTarget = false
	board := ui.board
	playerNum := ui.board.NextPlayer

	// Find off-board pieces that are placeable:
	offBoardPieces := generics.MakeSet[state.PieceType]()
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

	ui.selections.sourceOnBoardPons = nil
	for _, action := range board.Derived.Actions {
		if !action.Move {
			continue
		}
		currentPieces := ui.piecesOnBoard[action.SourcePos]
		if len(currentPieces) == 0 {
			klog.Errorf("No piece on UI in position %s, but valid action has that position as source!?", action.SourcePos)
			continue
		}
		pons := currentPieces[len(currentPieces)-1]
		if pons.PieceType != action.Piece || pons.Player != playerNum {
			klog.Errorf("Piece on UI in %s (Player #%d, %s) doesn't match not the valid action source (Player #%d, %s)!?",
				action.SourcePos, pons.Player+1, pons.PieceType, playerNum+1, action.Piece)
			continue
		}
		ui.selections.sourceOnBoardPons = append(ui.selections.sourceOnBoardPons, pons)
		ui.makeSelectable(pons)
	}

	// Reshapes as needed.
	ui.AdjustSelections()
}

// resetSourceSelection will unmake sources as selectable -- because we are moving to selecting the target.
func (ui *WebUI) resetSourceSelection() {
	for _, pons := range ui.selections.sourceOffBoardPons {
		ui.makeUnselectable(pons)
	}
	for _, pons := range ui.selections.sourceOnBoardPons {
		ui.makeUnselectable(pons)
	}
	ui.selections.sourceOffBoardPons = nil
}

func (ui *WebUI) selectTarget() {
	ui.selections.isSelectingSource = false
	ui.selections.isSelectingTarget = true
	ui.resetSourceSelection()

	ui.selections.targetActions = make([]state.Action, 0, len(ui.board.Derived.Actions))
	if ui.selections.selectedSourceIsOffBoard {
		// Placing a new piece.
		for _, action := range ui.board.Derived.Actions {
			if action.Move {
				continue
			}
			if action.Piece != ui.selections.selectedSourcePons.PieceType {
				continue
			}
			ui.selections.targetActions = append(ui.selections.targetActions, action)
		}
	} else {
		// Moving piece.
		for _, action := range ui.board.Derived.Actions {
			if !action.Move {
				continue
			}
			if !action.SourcePos.Equal(ui.selections.selectedSourcePos) {
				continue
			}
			ui.selections.targetActions = append(ui.selections.targetActions, action)
		}
	}

	// Make the source piece selectable: if selected the source is unselected.
	ui.makeSelectable(ui.selections.selectedSourcePons)

	// Notice even if we have only one target position to choose from, we still want to show and
	// ask the user to click on it to confirm the move -- the user may decide to change the source.
	playerNum := ui.board.NextPlayer
	ui.selections.targetPons = make([]*PieceOnScreen, len(ui.selections.targetActions))
	for ii, action := range ui.selections.targetActions {
		pos := action.TargetPos
		stackPos := len(ui.piecesOnBoard[pos])
		pons := &PieceOnScreen{
			Index:     ui.piecesOnBoardIdx,
			Player:    playerNum,
			PieceType: action.Piece,
			StackPos:  stackPos,
			Hex: svg.SVGPolygonElementFromWrapper(CreateSVG("polygon", Attrs{
				"stroke":       "url(#reliefStroke)",
				"stroke-width": ui.strokeWidth() * ui.Scale,
				"fill":         "black",
				"fill-opacity": 0.3,
			})),
		}
		pons.Hex.SetOnMouseUp(func(event *htmlevent.MouseEvent, currentTarget *svg.SVGElement) {
			ui.OnSelectTarget(ii)
		})

		ui.MovePieceTo(pons, pos, stackPos)
		ui.makeSelectable(pons)
		ui.insertOnBoardPieceIntoDOM(pons)
		ui.selections.targetPons[ii] = pons
	}

	ui.AdjustOffBoardPieces()
	ui.AdjustOnBoardPieces()
	ui.AdjustSelections()
}

func (ui *WebUI) resetTargetSelection() {
	for _, pons := range ui.selections.targetPons {
		ui.makeUnselectable(pons)
		pons.Hex.Remove()

	}
	ui.makeUnselectable(ui.selections.selectedSourcePons)
	ui.selections.targetPons = nil
	ui.selections.targetActions = nil
	ui.selections.isSelectingTarget = false
	ui.AdjustOffBoardPieces()
	ui.AdjustOnBoardPieces()
	ui.AdjustSelections()
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
		// Undo source selection: stop target selection and go back to source selection.
		ui.resetTargetSelection()
		ui.selections.selectedSourcePons = nil
		ui.selectSource()
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
	//fmt.Printf("Selected off-board %s (player #%d)\n", pons.PieceType, pons.Player+1)
	ui.resetSourceSelection()
	ui.selectTarget()
}

func (ui *WebUI) cancelSelection() {
	if !ui.selections.isSelecting {
		return
	}
	if ui.selections.isSelectingTarget {
		ui.resetTargetSelection()
		ui.selections.selectedSourcePons = nil
		ui.selectSource()
		return
	} else if ui.selections.isSelectingSource {
		// Nothing was selected yet, no-op.
	}
}

// OnSelectOnBoardPiece is called when an on-board piece is clicked.
func (ui *WebUI) OnSelectOnBoardPiece(pons *PieceOnScreen, pos state.Pos) {
	if !ui.selections.isSelecting {
		return
	}
	if ui.selections.isSelectingTarget {
		if pons != ui.selections.selectedSourcePons {
			return
		}
		// Undo source selection: stop target selection and go back to source selection.
		ui.resetTargetSelection()
		ui.selections.selectedSourcePons = nil
		ui.selectSource()
		return
	}

	var found bool
	for _, srcPons := range ui.selections.sourceOnBoardPons {
		if srcPons == pons {
			found = true
			break
		}
	}
	if !found {
		return
	}
	ui.selections.selectedSourceIsOffBoard = false
	ui.selections.selectedSourcePons = pons
	ui.selections.selectedSourcePos = pos
	ui.resetSourceSelection()
	ui.selectTarget()
}

func (ui *WebUI) OnSelectTarget(targetIdx int) {
	action := ui.selections.targetActions[targetIdx]
	ui.resetTargetSelection()
	ui.selections.selectedAction <- action
}
