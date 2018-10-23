package main

import (
	"fmt"

	"github.com/gopherjs/gopherjs/js"
	"github.com/gopherjs/jquery"
	"github.com/janpfeifer/hiveGo/state"
)

var (
	// Patterns for pieces.
	boardPiecesPatterns []*js.Object
	boardPiecesImages   []*js.Object

	// Map of all pieces currently on board.
	piecesOnBoard = make(map[state.Pos][]*PieceOnBoard)
)

type PieceOnBoard struct {
	Hex, Rect *js.Object
}

func PieceToPatternId(p state.Piece) string {
	return fmt.Sprintf("board_%s", state.PieceNames[p])
}

func init() {
	for ii := state.ANT; ii < state.LAST_PIECE_TYPE; ii++ {
		pattern := CreateSVG("pattern", Attrs{
			"id":           PieceToPatternId(ii),
			"patternUnits": "objectBoundingBox",
			"width":        "1.0",
			"height":       "1.0",
			"x":            "0",
			"y":            "0",
		})
		boardPiecesPatterns = append(boardPiecesPatterns, pattern)
		image := CreateSVG("image", Attrs{
			"href": fmt.Sprintf(
				"/github.com/janpfeifer/hiveGo/images/%s.png",
				state.PieceNames[ii]),
			"width":  1024,
			"height": 1024,
		})
		boardPiecesImages = append(boardPiecesImages, image)
		jq(pattern).Append(image)
		svgDefs.Append(pattern)
	}
}

func (pob *PieceOnBoard) MoveTo(pos state.Pos, stackPos int) {
	xc, yc := ui.PosToXY(pos, stackPos)
	face := ui.Face()

	// Move hexagon around piece: six points in a polygon, start
	// on left corner and move clockwise.
	const POINT_FORMAT = "%f,%f "
	const HEX_POINTS_FORMAT = POINT_FORMAT + POINT_FORMAT +
		POINT_FORMAT + POINT_FORMAT + POINT_FORMAT + POINT_FORMAT
	height := hexTriangleHeight(face)
	attrs := Attrs{
		"points": fmt.Sprintf(HEX_POINTS_FORMAT,
			xc-face, yc,
			xc-face/2.0, yc-height,
			xc+face/2.0, yc-height,
			xc+face, yc,
			xc+face/2.0, yc+height,
			xc-face/2.0, yc+height)}
	SetAttrs(pob.Hex, attrs)

	// Move rectangle.
	rectSize := face * 1.22
	attrs = Attrs{
		"x": xc - rectSize/2.0, "y": yc - rectSize/2.0,
		"width": rectSize, "height": rectSize,
	}
	SetAttrs(pob.Rect, attrs)
}

func OnChangeOfUIParams() {
	// Scale papterns.
	scale := 0.04 * ui.Scale
	for _, image := range boardPiecesImages {
		SetAttrs(image, Attrs{
			"width":  scale * 1024,
			"height": scale * 1024,
		})
	}

	// Scale hexagons.
	for pos, slice := range piecesOnBoard {
		for stackPos, pob := range slice {
			pob.MoveTo(pos, stackPos)
			SetAttrs(pob.Hex, Attrs{
				"stroke-width": 2.0 * ui.Scale,
			})
		}
	}
}

func Place(player int, action state.Action) {
	pos := action.TargetPos
	stack := piecesOnBoard[pos]
	pob := &PieceOnBoard{
		Hex: CreateSVG("polygon", Attrs{
			"stroke":       "black",
			"stroke-width": 2.0 * ui.Scale,
			"fill-opacity": 0.0,
		}),
		Rect: CreateSVG("rect", Attrs{
			"stroke":       "black",
			"stroke-width": 1.0,
			"fill":         fmt.Sprintf("url(#%s)", PieceToPatternId(action.Piece)),
		}),
	}
	pob.MoveTo(pos, len(stack))
	piecesOnBoard[pos] = append(stack, pob)

	canvas.Append(pob.Hex)
	canvas.Append(pob.Rect)
	jq(pob.Rect).On(jquery.MOUSEUP, func(e jquery.Event) {
		fmt.Printf("IsPropagationStopped=%v\n", e.IsPropagationStopped())
	})
	jq(pob.Hex).On(jquery.MOUSEUP, func(e jquery.Event) {
		// HexMoveTo(hex, 200, 200, 15)
		Alert("Click!")
	})
}
