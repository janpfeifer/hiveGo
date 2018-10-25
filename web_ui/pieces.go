package main

import (
	"fmt"

	"github.com/gopherjs/gopherjs/js"
	"github.com/gopherjs/jquery"
	"github.com/janpfeifer/hiveGo/state"
)

var (
	// Patterns for pieces.
	boardPiecesPatterns    []*js.Object
	boardPiecesImages      []*js.Object
	OffBoardPiecesPatterns []*js.Object
	OffBoardPiecesImages   []*js.Object

	// Map of all pieces currently on board.
	piecesOnBoard = make(map[state.Pos][]*PieceOnBoard)
)

const PieceDrawingScale = 1.2

type PieceOnBoard struct {
	Player    int
	Piece     state.Piece
	Hex, Rect *js.Object
}

const (
	BOARD    = "board_"
	OFFBOARD = "offboard_"
)

func PieceToPatternId(prefix string, p state.Piece) string {
	return prefix + state.PieceNames[p]
}

func init() {
	boardPiecesPatterns, boardPiecesImages =
		createPiecesPatternsAndImages(BOARD)
	OffBoardPiecesPatterns, OffBoardPiecesImages =
		createPiecesPatternsAndImages(OFFBOARD)
}

func createPiecesPatternsAndImages(prefix string) (patterns []*js.Object, images []*js.Object) {
	for ii := state.ANT; ii < state.LAST_PIECE_TYPE; ii++ {
		pattern := CreateSVG("pattern", Attrs{
			"id":           PieceToPatternId(prefix, ii),
			"patternUnits": "objectBoundingBox",
			"width":        "1.0",
			"height":       "1.0",
			"x":            "0",
			"y":            "0",
		})
		patterns = append(patterns, pattern)
		image := CreateSVG("image", Attrs{
			"href": fmt.Sprintf(
				"/github.com/janpfeifer/hiveGo/images/%s.png",
				state.PieceNames[ii]),
			"width":  1024,
			"height": 1024,
		})
		images = append(images, image)
		jq(pattern).Append(image)
		svgDefs.Append(pattern)
	}
	return patterns, images
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
	//rectSize := face * 1.22
	rectSize := face * 1.22 * PieceDrawingScale
	attrs = Attrs{
		"x": xc - rectSize/2.0, "y": yc - rectSize/2.0,
		"width": rectSize, "height": rectSize,
	}
	SetAttrs(pob.Rect, attrs)
}

func PiecesOnChangeOfUIParams() {
	// Scale papterns.
	//scale := 0.04 * ui.Scale
	scale := 0.04 * ui.Scale * PieceDrawingScale
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
	var color string
	if player == 0 {
		color = "cornsilk"
	} else {
		color = "darkkhaki"
	}
	pob := &PieceOnBoard{
		Player: player,
		Piece:  action.Piece,
		Hex: CreateSVG("polygon", Attrs{
			"stroke":       "black",
			"stroke-width": 2.0 * ui.Scale,
			"fill":         color,
			"fill-opacity": 1.0,
		}),
		Rect: CreateSVG("rect", Attrs{
			"stroke":         "black",
			"stroke-width":   0,
			"fill":           fmt.Sprintf("url(#%s)", PieceToPatternId(BOARD, action.Piece)),
			"pointer-events": "none",
		}),
	}
	pob.MoveTo(pos, len(stack))
	piecesOnBoard[pos] = append(stack, pob)

	BoardGroup.Append(pob.Hex)
	BoardGroup.Append(pob.Rect)
	jq(pob.Rect).On(jquery.MOUSEUP, func(e jquery.Event) {
		fmt.Printf("IsPropagationStopped=%v\n", e.IsPropagationStopped())
	})
	jq(pob.Hex).On(jquery.MOUSEUP, func(e jquery.Event) {
		// HexMoveTo(hex, 200, 200, 15)
		Alert("Click!")
	})
}
