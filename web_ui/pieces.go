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
	piecesOnBoard = make(map[state.Pos][]*PieceOnScreen)
)

const (
	// Interface visual constants.
	PIECE_DRAWING_SCALE = 1.464
	HEX_STROKE_WIDTH    = 2.0
	IMAGE_BASE_SIZE     = 0.0488
)

type PieceOnScreen struct {
	Index     int
	Player    uint8
	Piece     state.Piece
	Hex, Rect jquery.JQuery
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
			"x":            "-0.040",
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

func moveHexToXYFace(hex *js.Object, xc, yc, face float64) {
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
	SetAttrs(hex, attrs)
}

func (pons *PieceOnScreen) moveToXYFace(xc, yc, face float64) {
	moveHexToXYFace(Obj(pons.Hex), xc, yc, face)

	// Move rectangle.
	//rectSize := face * 1.22
	rectSize := face * PIECE_DRAWING_SCALE
	attrs := Attrs{
		"x": xc - rectSize/2.0, "y": yc - rectSize/2.0,
		"width": rectSize, "height": rectSize,
	}
	SetAttrs(Obj(pons.Rect), attrs)
}

func (pons *PieceOnScreen) MoveTo(pos state.Pos, stackPos int) {
	xc, yc := ui.PosToXY(pos, stackPos)
	face := ui.Face()
	pons.moveToXYFace(xc, yc, face)
}

func (pons *PieceOnScreen) OffBoardXYFace(stackPos int) (xc, yc, face float64) {
	face = STANDARD_FACE * ui.PixelRatio
	xc = (float64(pons.Piece)-float64(state.GRASSHOPPER))*4*face + float64(ui.Width)/2
	offBoardHeight := float64(ui.OffBoardHeight())
	yc = offBoardHeight / 2.0
	if pons.Player == 1 {
		yc = float64(ui.Height) - yc
	}

	// Stack effect.
	xc += float64(stackPos) * 3.0 * ui.PixelRatio
	yc -= float64(stackPos) * 3.0 * ui.PixelRatio
	return
}

func (pons *PieceOnScreen) OffBoardMove(stackPos int) {
	xc, yc, face := pons.OffBoardXYFace(stackPos)
	pons.moveToXYFace(xc, yc, face)
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
		for stackPos, pons := range slice {
			pons.MoveTo(pos, stackPos)
			SetAttrs(Obj(pons.Hex), Attrs{
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
	pons := &PieceOnScreen{
		Player: player,
		Piece:  action.Piece,
		Hex: jq(CreateSVG("polygon", Attrs{
			"stroke":       "black",
			"stroke-width": HEX_STROKE_WIDTH * ui.Scale,
			"fill":         PlayerBackgroundColor(player),
			"fill-opacity": 1.0,
		})),
		Rect: jq(CreateSVG("rect", Attrs{
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

	BoardGroup.Append(pons.Hex)
	BoardGroup.Append(pons.Rect)
	pons.Hex.On(jquery.MOUSEUP, func(e jquery.Event) {
		pons.OnSelectOnBoard(pos)
	})
}

func RemovePiece(player uint8, action state.Action) {
	pos := action.SourcePos
	stack := piecesOnBoard[pos]
	if len(stack) == 0 {
		fmt.Printf("Invalid move, there are no pieces in %s, action=%s!?\n", pos, action)
	}
	pons := stack[len(stack)-1]
	if len(stack) == 1 {
		delete(piecesOnBoard, pos)
	} else {
		// Pop piece from the top.
		piecesOnBoard[pos] = stack[0 : len(stack)-1]

	}
	pons.Hex.Remove()
	pons.Rect.Remove()
}

var (
	piecesOffBoard [state.NUM_PLAYERS]map[state.Piece][]*PieceOnScreen
)

func PlaceOffBoardPieces(b *state.Board) {
	index := 0
	for player := uint8(0); player < state.NUM_PLAYERS; player++ {
		piecesOffBoard[player] = make(map[state.Piece][]*PieceOnScreen)
		for piece := state.ANT; piece < state.LAST_PIECE_TYPE; piece++ {
			num_pieces := b.Available(player, piece)
			pieces := make([]*PieceOnScreen, 0, num_pieces)
			for stack := 0; stack < int(num_pieces); stack++ {
				pons := &PieceOnScreen{
					Index:  index,
					Player: player,
					Piece:  piece,
					Hex: jq(CreateSVG("polygon", Attrs{
						"stroke":       "black",
						"stroke-width": HEX_STROKE_WIDTH * ui.PixelRatio,
						"fill":         PlayerBackgroundColor(player),
						"fill-opacity": 1.0,
					})),
					Rect: jq(CreateSVG("rect", Attrs{
						"stroke":         "black",
						"stroke-width":   0,
						"fill":           fmt.Sprintf("url(#%s)", PieceToPatternId(OFFBOARD, piece)),
						"pointer-events": "none",
					})),
				}
				index++
				OffBoardGroups[player].Append(pons.Hex)
				OffBoardGroups[player].Append(pons.Rect)
				pons.Hex.On(jquery.MOUSEUP, func(e jquery.Event) {
					pons.OnSelectOffBoard()
				})
				pieces = append(pieces, pons)
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
			for stackPos, pons := range pieces {
				pons.OffBoardMove(stackPos)
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

func RemoveOffBoardPiece(player uint8, action state.Action) {
	stackPos := len(piecesOffBoard[player][action.Piece]) - 1
	pons := piecesOffBoard[player][action.Piece][stackPos]
	piecesOffBoard[player][action.Piece] = piecesOffBoard[player][action.Piece][0:stackPos]
	pons.Hex.Remove()
	pons.Rect.Remove()
}
