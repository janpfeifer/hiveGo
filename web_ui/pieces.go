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

const (
	// Interface visual constants.
	PIECE_DRAWING_SCALE = 1.464
	HEX_STROKE_WIDTH    = 2.5
	IMAGE_BASE_SIZE     = 0.0488
)

type PieceOnBoard struct {
	Player    uint8
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

func (pob *PieceOnBoard) moveToXYFace(xc, yc, face float64, stackPos int) {
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
	rectSize := face * PIECE_DRAWING_SCALE
	attrs = Attrs{
		"x": xc - rectSize/2.0, "y": yc - rectSize/2.0,
		"width": rectSize, "height": rectSize,
	}
	SetAttrs(pob.Rect, attrs)
}

func (pob *PieceOnBoard) MoveTo(pos state.Pos, stackPos int) {
	xc, yc := ui.PosToXY(pos, stackPos)
	face := ui.Face()
	pob.moveToXYFace(xc, yc, face, stackPos)
}

func (pob *PieceOnBoard) OffBoardMove(stackPos int) {
	face := STANDARD_FACE * ui.PixelRatio
	xc := (float64(pob.Piece)-float64(state.GRASSHOPPER))*4*face + float64(ui.Width)/2
	offBoardHeight := float64(ui.OffBoardHeight())
	yc := offBoardHeight / 2.0
	if pob.Player == 1 {
		yc = float64(ui.Height) - yc
	}

	// Stack effect.
	xc += float64(stackPos) * 3.0 * ui.PixelRatio
	yc -= float64(stackPos) * 3.0 * ui.PixelRatio

	pob.moveToXYFace(xc, yc, face, stackPos)
}

func PiecesOnChangeOfUIParams() {
	// Scale papterns.
	scale := IMAGE_BASE_SIZE * ui.Scale
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
				"stroke-width": HEX_STROKE_WIDTH * ui.Scale,
			})
		}
	}
}

func PlayerBackgroundColor(player uint8) string {
	if player == 0 {
		return "cornsilk"
	} else {
		return "darkkhaki"
	}

}

func Place(player uint8, action state.Action) {
	pos := action.TargetPos
	stack := piecesOnBoard[pos]
	pob := &PieceOnBoard{
		Player: player,
		Piece:  action.Piece,
		Hex: CreateSVG("polygon", Attrs{
			"stroke":       "black",
			"stroke-width": HEX_STROKE_WIDTH * ui.Scale,
			"fill":         PlayerBackgroundColor(player),
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

var (
	piecesOffBoard [state.NUM_PLAYERS]map[state.Piece][]*PieceOnBoard
)

func PlaceOffBoardPieces(b *state.Board) {
	for player := uint8(0); player < state.NUM_PLAYERS; player++ {
		piecesOffBoard[player] = make(map[state.Piece][]*PieceOnBoard)
		for piece := state.ANT; piece < state.LAST_PIECE_TYPE; piece++ {
			num_pieces := b.Available(player, piece)
			pieces := make([]*PieceOnBoard, 0, num_pieces)
			for stack := 0; stack < int(num_pieces); stack++ {
				pob := &PieceOnBoard{
					Player: player,
					Piece:  piece,
					Hex: CreateSVG("polygon", Attrs{
						"stroke":       "black",
						"stroke-width": HEX_STROKE_WIDTH * ui.PixelRatio,
						"fill":         PlayerBackgroundColor(player),
						"fill-opacity": 1.0,
					}),
					Rect: CreateSVG("rect", Attrs{
						"stroke":         "black",
						"stroke-width":   0,
						"fill":           fmt.Sprintf("url(#%s)", PieceToPatternId(OFFBOARD, piece)),
						"pointer-events": "none",
					}),
				}
				OffBoardGroups[player].Append(pob.Hex)
				OffBoardGroups[player].Append(pob.Rect)
				pieces = append(pieces, pob)
			}
			piecesOffBoard[player][piece] = pieces
		}
	}
}

func AdjustOffBoardPieces() {
	// Adjust pieces positions.
	for player := uint8(0); player < state.NUM_PLAYERS; player++ {
		for piece := state.ANT; piece < state.LAST_PIECE_TYPE; piece++ {
			pieces := piecesOffBoard[player][piece]
			for stackPos, pob := range pieces {
				pob.OffBoardMove(stackPos)
			}
		}
	}

	// Adjust pattern sizes.
	scale := IMAGE_BASE_SIZE * ui.PixelRatio
	for _, image := range OffBoardPiecesImages {
		SetAttrs(image, Attrs{
			"width":  scale * 1024,
			"height": scale * 1024,
		})
	}
}
