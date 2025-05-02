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

const (
	BOARD    = "board_"
	OFFBOARD = "offboard_"
)

func PieceToPatternId(prefix string, p state.PieceType) string {
	return prefix + state.PieceNames[p]
}

func init() {
	boardPiecesPatterns, boardPiecesImages =
		createPiecesPatternsAndImages(BOARD)
	OffBoardPiecesPatterns, OffBoardPiecesImages =
		createPiecesPatternsAndImages(OFFBOARD)
}

func createPiecesPatternsAndImages(prefix string) (patterns []*js.Object, images []*js.Object) {
	for ii := state.ANT; ii < state.LastPiece; ii++ {
		pattern := svg
		pattern := Old2.CreateSVG("pattern", Old2.Attrs{
			"id":           PieceToPatternId(prefix, ii),
			"patternUnits": "objectBoundingBox",
			"width":        "1.0",
			"height":       "1.0",
			"x":            "-0.040",
			"y":            "0",
		})
		patterns = append(patterns, pattern)
		image := Old2.CreateSVG("image", Old2.Attrs{
			"href": fmt.Sprintf(
				"/github.com/janpfeifer/hiveGo/images/%s.png",
				state.PieceNames[ii]),
			"width":  1024,
			"height": 1024,
		})
		images = append(images, image)
		jq(pattern).Append(image)
		SvgDefs.Append(pattern)
	}
	return patterns, images
}

func moveHexToXYFace(hex *js.Object, xc, yc, face float64) {
	// Move hexagon around the piece: six points in a polygon, start
	// on the left corner and move clockwise.
	const PointFormat = "%f,%f "
	const HexPointsFormat = PointFormat + PointFormat +
		PointFormat + PointFormat + PointFormat + PointFormat
	height := Old2.hexTriangleHeight(face)
	attrs := Old2.Attrs{
		"points": fmt.Sprintf(HexPointsFormat,
			xc-face, yc,
			xc-face/2.0, yc-height,
			xc+face/2.0, yc-height,
			xc+face, yc,
			xc+face/2.0, yc+height,
			xc-face/2.0, yc+height)}
	Old2.SetAttrs(hex, attrs)
}

func (pons *PieceOnScreen) moveToXYFace(xc, yc, face float64) {
	moveHexToXYFace(Old2.Obj(pons.Hex), xc, yc, face)

	// Move rectangle.
	rectSize := face * PieceDrawingScale
	attrs := Old2.Attrs{
		"x": xc - rectSize/2.0, "y": yc - rectSize/2.0,
		"width": rectSize, "height": rectSize,
	}
	Old2.SetAttrs(Old2.Obj(pons.Rect), attrs)
}

func (pons *PieceOnScreen) MoveTo(pos state.Pos, stackPos int) {
	xc, yc := ui.PosToXY(pos, stackPos)
	face := ui.Face()
	pons.moveToXYFace(xc, yc, face)
}

func (pons *PieceOnScreen) OffBoardXYFace(stackPos int) (xc, yc, face float64) {
	face = Old2.StandardFace * ui.PixelRatio
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
	// Scale the patterns.
	scale := ImageBaseSize * ui.Scale
	for _, image := range boardPiecesImages {
		Old2.SetAttrs(image, Old2.Attrs{
			"width":  scale * 1024,
			"height": scale * 1024,
		})
	}

	// Scale hexagons.
	for pos, slice := range piecesOnBoard {
		for stackPos, pons := range slice {
			pons.MoveTo(pos, stackPos)
			Old2.SetAttrs(Old2.Obj(pons.Hex), Old2.Attrs{
				"stroke-width": HexStrokeWidth * ui.Scale,
			})
		}
	}
}

func PlayerBackgroundColor(player state.PlayerNum) string {
	if player == state.PlayerFirst {
		return "cornsilk"
	} else {
		return "darkkhaki"
	}

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

func RemovePiece(action state.Action) {
	pos := action.SourcePos
	stack := piecesOnBoard[pos]
	if len(stack) == 0 {
		fmt.Printf("Invalid move, there are no pieces in %s, action=%s!?\n", pos, action)
	}
	pons := stack[len(stack)-1]
	if len(stack) == 1 {
		delete(piecesOnBoard, pos)
	} else {
		// Pop the top piece.
		piecesOnBoard[pos] = stack[0 : len(stack)-1]

	}
	pons.Hex.Remove()
	pons.Rect.Remove()
}

var (
	piecesOffBoard [state.NumPlayers]map[state.PieceType][]*PieceOnScreen
)

func (g *Old2.Game) PlaceOffBoardPieces() {
	index := 0
	for player := state.PlayerNum(0); player < state.NumPlayers; player++ {
		piecesOffBoard[player] = make(map[state.PieceType][]*PieceOnScreen)
		for piece := state.ANT; piece < state.LastPiece; piece++ {
			numPieces := g.board.Available(player, piece)
			pieces := make([]*PieceOnScreen, 0, numPieces)
			for stack := 0; stack < int(numPieces); stack++ {
				pons := &PieceOnScreen{
					Index:  index,
					Player: player,
					Piece:  piece,
					Hex: jq(Old2.CreateSVG("polygon", Old2.Attrs{
						"stroke":       "black",
						"stroke-width": HexStrokeWidth * ui.PixelRatio,
						"fill":         PlayerBackgroundColor(player),
						"fill-opacity": 1.0,
					})),
					Rect: jq(Old2.CreateSVG("rect", Old2.Attrs{
						"stroke":         "black",
						"stroke-width":   0,
						"fill":           fmt.Sprintf("url(#%s)", PieceToPatternId(OFFBOARD, piece)),
						"pointer-events": "none",
					})),
				}
				index++
				Old2.OffBoardGroups[player].Append(pons.Hex)
				Old2.OffBoardGroups[player].Append(pons.Rect)
				pons.Hex.On(jquery.MOUSEUP, func(e jquery.Event) {
					g.OnSelectOffBoard(pons)
				})
				pieces = append(pieces, pons)
			}
			piecesOffBoard[player][piece] = pieces
		}
	}
}

func AdjustOffBoardPieces() {
	// Adjust pieces positions.
	for player := range state.PlayerNum(2) {
		for piece := state.ANT; piece < state.LastPiece; piece++ {
			pieces := piecesOffBoard[player][piece]
			for stackPos, pons := range pieces {
				pons.OffBoardMove(stackPos)
			}
		}
	}

	// Adjust pattern sizes.
	scale := ImageBaseSize * ui.PixelRatio
	for _, image := range OffBoardPiecesImages {
		Old2.SetAttrs(image, Old2.Attrs{
			"width":  scale * 1024,
			"height": scale * 1024,
		})
	}
}

func RemoveOffBoardPiece(player state.PlayerNum, action state.Action) {
	stackPos := len(piecesOffBoard[player][action.Piece]) - 1
	pons := piecesOffBoard[player][action.Piece][stackPos]
	piecesOffBoard[player][action.Piece] = piecesOffBoard[player][action.Piece][0:stackPos]
	pons.Hex.Remove()
	pons.Rect.Remove()
}
