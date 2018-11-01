package main

import (
	"fmt"

	"github.com/gopherjs/jquery"
	"github.com/janpfeifer/hiveGo/state"
)

var (
	// Hex displayed around selected pieces, either on or offboard,
	// and the corresponding available target positions.
	SourceOnBoardPOnS, SourceOffBoardPOnS *PieceOnScreen
	SourceOnBoardHex, SourceOffBoardHex   jquery.JQuery
	TargetHexes                           []jquery.JQuery
	TargetActions                         []state.Action
)

func Unselect() {
	if SourceOffBoardPOnS != nil {
		SourceOffBoardHex.Remove()
		SourceOffBoardPOnS = nil
	}

	if SourceOnBoardPOnS != nil {
		SourceOnBoardHex.Remove()
		SourceOnBoardPOnS = nil
	}

	// Remove target indicators and actions.
	for _, tgtHex := range TargetHexes {
		tgtHex.Remove()
	}
	TargetHexes = nil
	TargetActions = nil
}

func (pons *PieceOnScreen) OnSelectOffBoard() {
	if !IsRunning {
		return
	}

	// Make sure to take the top of the stack in case more
	// than one is selected.
	pieces := piecesOffBoard[pons.Player][pons.Piece]
	stackPos := len(pieces) - 1
	pieces[stackPos].stackTopOnSelectOffBoard(stackPos)
}

func (pons *PieceOnScreen) stackTopOnSelectOffBoard(stackPos int) {
	deselect := SourceOffBoardPOnS == pons

	// First deselect previous selection, if any.
	Unselect()

	// If the wrong player's turn, then just deselect.
	if pons.Player != Board.NextPlayer {
		return
	}

	// User just unselected currently selected one. In this
	// case there is nothing new to select.
	if deselect {
		return
	}

	// Collects valid target positions and checks if piece
	// can actually be put into play.
	validTargets := make(map[state.Pos]state.Action)
	for _, action := range Board.Derived.Actions {
		if !action.Move && action.Piece == pons.Piece {
			validTargets[action.TargetPos] = action
		}
	}
	if len(validTargets) == 0 {
		fmt.Printf("Can not place %s\n", pons.Piece)
		return
	}

	// Select piece to put on board.
	SourceOffBoardPOnS = pons
	SourceOffBoardHex = jq(CreateSVG("polygon", Attrs{
		"stroke":         "darkviolet",
		"stroke-width":   2 * HEX_STROKE_WIDTH * ui.PixelRatio,
		"fill-opacity":   0,
		"pointer-events": "none",
	}))
	xc, yc, face := pons.OffBoardXYFace(stackPos)
	moveHexToXYFace(Obj(SourceOffBoardHex), xc, yc, face)
	OffBoardGroups[pons.Player].Append(SourceOffBoardHex)

	MarkTargetActions(validTargets)
}

// OnSelectOnBoard first picks the top piece of the stack selected.
func (pons *PieceOnScreen) OnSelectOnBoard(pos state.Pos) {
	if !IsRunning {
		return
	}

	// Make sure to take the top of the stack in case more
	// than one is selected.
	pieces := piecesOnBoard[pos]
	stackPos := len(pieces) - 1
	pieces[stackPos].stackTopOnSelectOnBoard(pos, stackPos)
}

func (pons *PieceOnScreen) stackTopOnSelectOnBoard(pos state.Pos, stackPos int) {
	deselect := SourceOnBoardPOnS == pons

	// First deselect previous selection, if any.
	Unselect()

	// If the wrong player's turn, then just deselect.
	if pons.Player != Board.NextPlayer {
		return
	}

	// User just deselected currently selected one. In this
	// case there is nothing new to select.
	if deselect {
		return
	}

	// Collects valid target positions and checks if piece
	// can actually be moved.
	validTargets := make(map[state.Pos]state.Action)
	for _, action := range Board.Derived.Actions {
		if action.Move && action.SourcePos == pos {
			validTargets[action.TargetPos] = action
		}
	}
	if len(validTargets) == 0 {
		return
	}

	// Select piece to put on board.
	SourceOnBoardPOnS = pons
	SourceOnBoardHex = jq(CreateSVG("polygon", Attrs{
		"stroke":         "darkviolet",
		"stroke-width":   2 * HEX_STROKE_WIDTH * ui.PixelRatio,
		"fill-opacity":   0,
		"pointer-events": "none",
	}))
	xc, yc := ui.PosToXY(pos, stackPos)
	face := ui.Face()
	moveHexToXYFace(Obj(SourceOnBoardHex), xc, yc, face)
	BoardGroup.Append(SourceOnBoardHex)

	MarkTargetActions(validTargets)
}

func FaceForTargetSelection() float64 {
	return ui.Face() - HEX_STROKE_WIDTH*ui.PixelRatio
}

func MarkTargetActions(validTargets map[state.Pos]state.Action) {
	// List valid target positions on board.
	TargetActions = nil
	for pos, action := range validTargets {
		TargetActions = append(TargetActions, action)
		stackPos := len(piecesOnBoard[pos])
		xc, yc := ui.PosToXY(pos, stackPos)
		face := FaceForTargetSelection()
		hex := jq(CreateSVG("polygon", Attrs{
			"stroke":       "darkviolet",
			"stroke-width": 1.5 * HEX_STROKE_WIDTH * ui.PixelRatio,
			"stroke-dasharray": fmt.Sprintf("%d,%d",
				int(HEX_STROKE_WIDTH*ui.PixelRatio),
				int(HEX_STROKE_WIDTH*ui.PixelRatio)),
			"fill-opacity": 0,
		}))
		moveHexToXYFace(Obj(hex), xc, yc, face)
		BoardGroup.Append(hex)
		TargetHexes = append(TargetHexes, hex)
		thisAction := action // Local copy of action for closure below.
		jq(hex).On(jquery.MOUSEUP, func(e jquery.Event) {
			OnTargetClick(thisAction)
		})
	}
}

func SelectionsOnChangeOfUIParams() {
	// Source offboard selection.
	if SourceOffBoardPOnS != nil {
		pons := SourceOffBoardPOnS
		stackPos := len(piecesOffBoard[pons.Player][pons.Piece]) - 1
		xc, yc, face := pons.OffBoardXYFace(stackPos)
		moveHexToXYFace(Obj(SourceOffBoardHex), xc, yc, face)
	}

	// Candidate target actions.
	for ii, action := range TargetActions {
		stackPos := len(piecesOnBoard[action.TargetPos])
		xc, yc := ui.PosToXY(action.TargetPos, stackPos)
		face := FaceForTargetSelection()
		moveHexToXYFace(Obj(TargetHexes[ii]), xc, yc, face)
	}
}

func OnTargetClick(action state.Action) {
	Unselect()
	fmt.Printf("Selected action: %v\n", action)
	player := Board.NextPlayer

	if action.Move {
		RemovePiece(action)
	} else {
		RemoveOffBoardPiece(player, action)
	}

	// Place piece on target position.
	Place(player, action)

	// Finally execute action in the virtual board.
	ExecuteAction(action)
}
