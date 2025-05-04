package main

import (
	"fmt"
	"github.com/gowebapi/webapi/graphics/svg"
	"github.com/gowebapi/webapi/html/htmlevent"
	"github.com/janpfeifer/hiveGo/internal/state"
	"strconv"
	"strings"
)

const (
	// Interface visual constants:

	PieceDrawingScale = 1.464
	HexStrokeWidth    = 2.0
	ImageBaseSize     = 0.0488
)

// PieceOnScreen holds the information of piece being displayed on-screen.
type PieceOnScreen struct {
	Index     int
	Player    state.PlayerNum
	StackPos  int
	PieceType state.PieceType
	Hex       *svg.SVGPolygonElement
	Rect      *svg.SVGRectElement
}

// PieceLocation can be onBoard or offBoard.
type PieceLocation int8

const (
	OnBoard PieceLocation = iota
	OffBoard
)

// Face returns the size of a hexagon's face.
func (ui *WebUI) Face() float64 {
	return StandardFaceScale * ui.Scale
}

func PlayerBackgroundColor(player state.PlayerNum) string {
	if player == state.PlayerFirst {
		return "cornsilk"
	} else {
		return "darkkhaki"
	}
}

// ==================================================================================================================
// Crate pieces patterns  -------------------------------------------------------------------------------------------
// ==================================================================================================================

// OffBoardHeight for UI.
func (ui *WebUI) OffBoardHeight() int {
	return int(96.0 * ui.PixelRatio) // 128?
}

// pieceToPatternID returns the HTML ID for the given piece type.
func pieceToPatternID(loc PieceLocation, p state.PieceType) string {
	prefix := "onBoard"
	if loc == OffBoard {
		prefix = "offBoard"
	}
	return prefix + state.PieceNames[p]
}

// playerToTilePatternID returns the HTML ID for the given player tile.
func playerToTilePatternID(loc PieceLocation, player state.PlayerNum) string {
	prefix := "onBoard"
	if loc == OffBoard {
		prefix = "offBoard"
	}
	return strings.Join([]string{prefix, "Tile", strconv.Itoa(int(player))}, "")
}

// createPiecesPatternsAndImages for OnBoard or OffBoard.
func (ui *WebUI) createPiecesPatternsAndImages(loc PieceLocation) (patterns []*svg.SVGPatternElement, images []*svg.SVGImageElement) {
	for pieceType := state.ANT; pieceType < state.LastPiece; pieceType++ {
		pattern := svg.SVGPatternElementFromWrapper(CreateSVG("pattern", Attrs{
			"id":           pieceToPatternID(loc, pieceType),
			"patternUnits": "objectBoundingBox",
			"width":        "1.0",
			"height":       "1.0",
			"x":            "-0.040",
			"y":            "0",
		}))
		patterns = append(patterns, pattern)
		image := svg.SVGImageElementFromWrapper(CreateSVG("image", Attrs{
			"href": fmt.Sprintf(
				"./assets/%s.png",
				state.PieceNames[pieceType]),
			"width":  1024,
			"height": 1024,
		}))
		images = append(images, image)
		pattern.AppendChild(&image.Node)
		ui.defs.AppendChild(&pattern.Node)
	}
	return patterns, images
}

// ==================================================================================================================
// Place on-board pieces  -------------------------------------------------------------------------------------------
// ==================================================================================================================

// hexTriangleHeight returns the height of the triangles that make up for a hexagon, given the face length.
func hexTriangleHeight(face float64) float64 {
	return 0.866 * face // sqrt(3)/2 * face
}

// PosToXY converts board positions to screen position.
func (ui *WebUI) PosToXY(pos state.Pos, stackCount int) (x, y float64) {
	face := ui.Face()
	hexWidth := 1.5 * face
	triangleHeight := hexTriangleHeight(face)
	hexHeight := 2. * triangleHeight

	x = ui.ShiftX + float64(pos.X())*hexWidth
	y = ui.ShiftY + float64(pos.Y())*hexHeight
	if pos.X()%2 != 0 {
		y += triangleHeight
	}
	x += float64(stackCount) * 3.0 * ui.Scale
	y -= float64(stackCount) * 3.0 * ui.Scale

	// Add center of screen:
	x += float64(ui.Width / 2)
	y += float64(ui.Height / 2)
	return
}

func moveHexToXYFace(hex *svg.SVGPolygonElement, xc, yc, face float64) {
	// Move hexagon around the piece: six points in a polygon, start
	// on the left corner and move clockwise.
	const PointFormat = "%f,%f "
	const HexPointsFormat = PointFormat + PointFormat +
		PointFormat + PointFormat + PointFormat + PointFormat
	height := hexTriangleHeight(face)
	SetAttrs(&hex.Element, Attrs{
		"points": fmt.Sprintf(HexPointsFormat,
			xc-face, yc,
			xc-face/2.0, yc-height,
			xc+face/2.0, yc-height,
			xc+face, yc,
			xc+face/2.0, yc+height,
			xc-face/2.0, yc+height)})
}

func (ui *WebUI) movePieceToXYFace(pons *PieceOnScreen, xc, yc, face float64) {
	moveHexToXYFace(pons.Hex, xc, yc, face)

	// Move rectangle.
	rectSize := face * PieceDrawingScale
	SetAttrs(&pons.Rect.Element,
		Attrs{
			"x": xc - rectSize/2.0, "y": yc - rectSize/2.0,
			"width": rectSize, "height": rectSize,
		})
}

// MovePieceTo a state.Pos, it is automatically converted to the canvas position on the browser.
func (ui *WebUI) MovePieceTo(pons *PieceOnScreen, pos state.Pos, stackPos int) {
	xc, yc := ui.PosToXY(pos, stackPos)
	face := ui.Face()
	ui.movePieceToXYFace(pons, xc, yc, face)
}

// ==================================================================================================================
// Place off-board pieces  ------------------------------------------------------------------------------------------
// ==================================================================================================================

// offBoardXYFace returns the canvas position for the piece on the stackPos stack position.
func (ui *WebUI) offBoardXYFace(pons *PieceOnScreen, stackPos int) (xc, yc, face float64) {
	face = StandardFaceScale * ui.PixelRatio
	xc = (float64(pons.PieceType)-float64(state.GRASSHOPPER))*4*face + float64(ui.Width)/2
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

func (ui *WebUI) offBoardMovePiece(pons *PieceOnScreen, stackPos int) {
	xc, yc, face := ui.offBoardXYFace(pons, stackPos)
	ui.movePieceToXYFace(pons, xc, yc, face)
}

// CreateOffBoardPieces display the pieces that are off-board, still
// to be placed on game.
func (ui *WebUI) CreateOffBoardPieces(board *state.Board) {
	index := 0
	for playerNum := range state.PlayerInvalid {
		if current := ui.piecesOffBoard[playerNum]; current != nil {
			// Remove previous pieces:
			// TODO: remove previous game pieces.
		}
		ui.piecesOffBoard[playerNum] = make(map[state.PieceType][]*PieceOnScreen)
		for pieceType := state.ANT; pieceType < state.LastPiece; pieceType++ {
			numPieces := board.Available(playerNum, pieceType)
			pieces := make([]*PieceOnScreen, 0, numPieces)
			for range numPieces {
				pons := &PieceOnScreen{
					Index:     index,
					Player:    playerNum,
					PieceType: pieceType,
					Hex: svg.SVGPolygonElementFromWrapper(CreateSVG("polygon", Attrs{
						"stroke":       "url(#reliefStroke)",
						"stroke-width": HexStrokeWidth * ui.PixelRatio,
						"fill":         fmt.Sprintf("url(#%s)", playerToTilePatternID(OffBoard, playerNum)),

						//"fill":         PlayerBackgroundColor(player),
						"fill-opacity": 1.0,
					})),
					Rect: svg.SVGRectElementFromWrapper(CreateSVG("rect", Attrs{
						"stroke":         "black",
						"stroke-width":   0,
						"fill":           fmt.Sprintf("url(#%s)", pieceToPatternID(OffBoard, pieceType)),
						"pointer-events": "none",
					})),
				}
				index++
				ui.offBoardGroups[playerNum].AppendChild(&pons.Hex.Node)
				ui.offBoardGroups[playerNum].AppendChild(&pons.Rect.Node)
				pons.Hex.SetOnMouseUp(func(event *htmlevent.MouseEvent, currentTarget *svg.SVGElement) {
					ui.OnSelectOffBoardPiece(pons)
				})
				pieces = append(pieces, pons)
			}
			ui.piecesOffBoard[playerNum][pieceType] = pieces
		}
	}
}

// OnSelectOffBoardPiece is called when an off-board piece is clicked.
func (ui *WebUI) OnSelectOffBoardPiece(pons *PieceOnScreen) {
	fmt.Printf("Selected off-board %s (%s)\n", pons.PieceType, pons.Player)
}

func (ui *WebUI) AdjustOffBoardPieces() {
	// Adjust pieces positions.
	for player := range state.PlayerInvalid {
		for pieceType := state.ANT; pieceType < state.LastPiece; pieceType++ {
			pieces := ui.piecesOffBoard[player][pieceType]
			for stackPos, pons := range pieces {
				ui.offBoardMovePiece(pons, stackPos)
			}
		}
	}

	// Adjust pattern sizes.
	basePatternSize := 1024 * ImageBaseSize * ui.PixelRatio * StandardFaceScale / 33.0
	pieceScale := int(basePatternSize)
	fmt.Printf("Rescaling off-board piece images to %d x %d\n", pieceScale, pieceScale)
	for _, image := range ui.offBoardPiecesImages {
		SetAttrs(&image.Element, Attrs{
			"width":  pieceScale,
			"height": pieceScale,
		})
	}

	// Adjust pattern sizes.
	tileScale := int(basePatternSize * 50.0 / 36.0)
	fmt.Printf("Rescaling off-board tile images to %d x %d\n", tileScale, tileScale)
	for _, image := range ui.offBoardTilesImages {
		SetAttrs(&image.Element, Attrs{
			"width":  tileScale,
			"height": tileScale,
		})
	}

}
