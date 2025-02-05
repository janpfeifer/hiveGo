package main

import (
	"fmt"
	"github.com/janpfeifer/hiveGo/images"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"log"
	"math"

	"github.com/gotk3/gotk3/cairo"
	"github.com/gotk3/gotk3/gtk"
)

var (
	pieceSurfaces     [LastPiece]*cairo.Surface
	pieceBaseSurfaces [NumPlayers]*cairo.Surface
)

// Standard face size.
const standardFace = 33.0

var (
	// Current zoom factor and translation (resulting from dragging board).
	zoomFactor     = 1.0
	shiftX, shiftY = 0.0, 0.0

	// Currently selected off-board piece (NoPiece if nothing is selected)
	selectedOffBoardPiece = NoPiece

	// Currently selected piece to move.
	hasSelectedPiece = false
	selectedPiecePos Pos
)

func loadImageResources() {
	_, err := images.CopyToTmpDir()
	if err != nil {
		log.Fatalf("Failed to create files for images: %+v", err)
	}
	// Load each piece drawing.
	for ii := PieceType(1); ii < LastPiece; ii++ {
		imagePath := images.NameToInfo[PieceNames[ii]].Path()
		if imagePath == "" {
			log.Fatalf("Cannot find image for piece %q", PieceNames[ii])
		}
		surface, err := cairo.NewSurfaceFromPNG(imagePath)
		if err != nil {
			log.Fatalf("Failed to read image for piece in %q: %v", imagePath, err)
		}
		pieceSurfaces[ii] = surface
	}

	// Load base piece surface for each player.
	for ii := 0; ii < NumPlayers; ii++ {
		imagePath := images.NameToInfo[fmt.Sprintf("tile_player_%d", ii)].Path()
		if imagePath == "" {
			log.Fatalf("Cannot find image for tile %d", ii)
		}
		surface, err := cairo.NewSurfaceFromPNG(imagePath)
		if err != nil {
			log.Fatalf("Failed to read %q, the tile for player %d: %+v", imagePath, ii, err)
		}
		// log.Printf("Loaded tile_player_%d.png\n", ii)
		pieceBaseSurfaces[ii] = surface
	}
}

// Parameters used to draw the main board.
type drawingParams struct {
	width, height       float64
	face                float64
	xc, yc              float64
	hexWidth, hexHeight float64
}

func newDrawingParams(da *gtk.DrawingArea) (dp *drawingParams) {
	allocation := da.GetAllocation()
	dp = &drawingParams{
		width:  float64(allocation.GetWidth()),
		height: float64(allocation.GetHeight()),
		face:   standardFace * zoomFactor,
	}
	dp.xc, dp.yc = dp.width/2.0+shiftX, dp.height/2.0+shiftY
	dp.hexWidth = 1.5 * dp.face
	dp.hexHeight = 2. * hexTriangleHeight(dp.face)
	return
}

func (dp *drawingParams) posToXY(pos Pos, stackCount int) (x, y float64) {
	//x = dp.xc + float64(pos.X())*dp.hexWidth
	//y = dp.yc + float64(pos.Y())*dp.hexHeight
	//if pos.X()%2 != 0 {
	//	y += hexTriangleHeight(dp.face)
	//}
	x = dp.xc + float64(pos.X())*dp.hexWidth
	y = dp.yc + (float64(pos.Y())+float64(pos.X())/2.0)*dp.hexHeight
	x += float64(stackCount) * 3.0 * zoomFactor
	y -= float64(stackCount) * 3.0 * zoomFactor
	return
}

func (dp *drawingParams) XYToPos(x, y float64) Pos {
	x -= dp.xc
	y -= dp.yc
	posX := int8(math.Round(x / dp.hexWidth))
	//if posX%2 != 0 {
	//	y -= hexTriangleHeight(dp.face)
	//}
	//posY := int8(math.Round(y / dp.hexHeight))
	posY := int8(math.Round(y/dp.hexHeight - float64(posX)/2.0))
	return Pos{posX, posY}
}

func drawMainBoard(da *gtk.DrawingArea, cr *cairo.Context) {
	cr.Save()
	defer cr.Restore()

	dp := newDrawingParams(da)
	drawBackground(da, cr, 244./256., 231./256., 210./256., true, 0.0)
	if !started {
		drawFullSurface(da, cr, pieceSurfaces[3])
		return
	}

	// Draw pieces on the board.
	face := standardFace * zoomFactor
	poss := board.OccupiedPositions()
	PosSort(poss)
	for _, pos := range poss {
		player, piece, stacked := board.PieceAt(pos)
		if !stacked {
			x, y := dp.posToXY(pos, 0)
			drawPieceAndBase(da, cr, player, piece, face, x, y)
		} else {
			stack := board.StackAt(pos)
			count := int(stack.CountPieces())
			for ii := 0; ii < count; ii++ {
				player, piece = stack.PieceAt(uint8(count - ii - 1))
				x, y := dp.posToXY(pos, ii)
				drawPieceAndBase(da, cr, player, piece, face, x, y)
			}
			// Draw small icons of pieces under the stack.
			x, y := dp.posToXY(pos, count-1)
			for ii := 0; ii < count-1; ii++ {
				idx := uint8(count - ii - 1)
				player, piece = stack.PieceAt(idx)
				drawPieceAndBase(da, cr, player, piece, face/5.0,
					x+(float64(ii)-2)*0.25*face, y+0.75*hexTriangleHeight(face))
			}
		}

	}

	// Draw placement candidates.
	if selectedOffBoardPiece != NoPiece {
		drawPlacementPositions(da, cr, dp)
	}

	// Draw piece selected to move.
	if hasSelectedPiece {
		drawMovePositions(da, cr, dp)
	}

	cr.Clip()
}

func drawPlacementPositions(da *gtk.DrawingArea, cr *cairo.Context, dp *drawingParams) {
	posMap := placementPositions()
	for pos := range posMap {
		drawHexagonBoardTarget(da, cr, dp, pos)
	}
}

func drawMovePositions(da *gtk.DrawingArea, cr *cairo.Context, dp *drawingParams) {
	drawHexagonBoardSelection(da, cr, dp, selectedPiecePos)
	for _, action := range board.Derived.Actions {
		if action.Move && action.SourcePos == selectedPiecePos {
			drawHexagonBoardTarget(da, cr, dp, action.TargetPos)
		}
	}
}

var rainbowColors = [][3]float64{
	{255. / 255., 0, 0},
	{255. / 255., 127. / 255., 0},
	{255. / 255., 255. / 255., 0},
	{0, 255. / 255., 0},
	{0, 0, 255. / 255.},
	{75. / 255., 0, 130. / 255.},
	{148. / 255., 0, 211. / 255.},
}

func drawOffBoardArea(da *gtk.DrawingArea, cr *cairo.Context, player uint8) {
	cr.Save()
	defer cr.Restore()

	// Background
	drawBackground(da, cr, 0.8, 0.8, 0.8, true, 0.0)
	if started && board.NextPlayer == player {
		drawBackground(da, cr, 0.6, 1.0, 0.6, false, 5.0)
	}
	if finished && board.Derived.Wins[player] {
		for ii, color := range rainbowColors {
			drawBackground(da, cr, color[0], color[1], color[2], false, float64(len(rainbowColors)-ii)*5.0)
		}
	}

	// Loop over the piece types
	for piece := PieceType(1); piece < LastPiece; piece++ {
		count := board.Available(player, piece)
		if count == 0 {
			continue
		}
		x, y := offBoardPieceToPosition(da, piece)
		for ii := 0; ii < int(count); ii++ {
			adjX, adjY := x+3.0*float64(ii), y-3.0*float64(ii)
			drawPieceAndBase(da, cr, player, piece, standardFace, adjX, adjY)
			if ii == int(count)-1 && player == board.NextPlayer && piece == selectedOffBoardPiece {
				drawHexagonSelection(da, cr, standardFace, adjX, adjY)
			}
		}
	}

	cr.Clip()
}

func offBoardPieceToPosition(da *gtk.DrawingArea, piece PieceType) (x, y float64) {
	allocation := da.GetAllocation()
	width, height := float64(allocation.GetWidth()), float64(allocation.GetHeight())
	xc, yc := width/2.0, height/2.0

	// Spacing between each available piece.
	const spacing = 3.0 * standardFace
	x, y = xc+(float64(piece)-3.0)*spacing, yc
	return
}

func offBoardPositionToPiece(da *gtk.DrawingArea, x, y float64) (piece PieceType) {
	for piece = PieceType(1); piece < LastPiece; piece++ {
		pX, pY := offBoardPieceToPosition(da, piece)
		if math.Abs(x-pX) < standardFace && math.Abs(y-pY) < hexTriangleHeight(standardFace) {
			return
		}
	}
	piece = NoPiece
	return
}

// hexTriangleHeight returns the height of the triangles that make up for an hexagon, given the face lenght.
func hexTriangleHeight(face float64) float64 {
	return 0.866 * face // sqrt(3)/2 * face
}

func drawBackground(da *gtk.DrawingArea, cr *cairo.Context, r, g, b float64, fill bool, lineWidth float64) {
	cr.Save()
	defer cr.Restore()

	allocation := da.GetAllocation()
	width, height := float64(allocation.GetWidth()), float64(allocation.GetHeight())
	cr.SetSourceRGB(r, g, b)
	cr.Rectangle(0.0, 0.0, width, height)
	if fill {
		cr.FillPreserve()
	} else {
		cr.SetLineWidth(lineWidth)
		cr.StrokePreserve()
	}
}

// drawHexagon will draw it with the given face length centered at xc, yc.
func drawHexagon(da *gtk.DrawingArea, cr *cairo.Context, face, xc, yc float64) {
	height := hexTriangleHeight(face)

	// Start on left corner and move clockwise.
	cr.MoveTo(xc-face, yc)
	cr.LineTo(xc-face/2.0, yc-height)
	cr.LineTo(xc+face/2.0, yc-height)
	cr.LineTo(xc+face, yc)
	cr.LineTo(xc+face/2.0, yc+height)
	cr.LineTo(xc-face/2.0, yc+height)
	cr.LineTo(xc-face, yc)
	cr.Stroke()
}

func drawHexagonBoardTarget(da *gtk.DrawingArea, cr *cairo.Context, dp *drawingParams, pos Pos) {
	cr.Save()
	defer cr.Restore()

	cr.SetLineWidth(3.5)
	cr.SetLineJoin(cairo.LINE_JOIN_ROUND)
	cr.SetSourceRGB(0.204, 0.914, 0.169)
	drawHexagonBoard(da, cr, dp, pos)
}

func drawHexagonBoardSelection(da *gtk.DrawingArea, cr *cairo.Context, dp *drawingParams, pos Pos) {
	cr.Save()
	defer cr.Restore()

	cr.SetLineWidth(5.0)
	cr.SetLineJoin(cairo.LINE_JOIN_ROUND)
	cr.SetSourceRGB(0.38, 0.114, 0.549)
	drawHexagonBoard(da, cr, dp, pos)
}

func drawHexagonBoard(da *gtk.DrawingArea, cr *cairo.Context, dp *drawingParams, pos Pos) {
	stack := board.StackAt(pos)
	count := int(stack.CountPieces())
	if count > 0 {
		count = count - 1
	}
	x, y := dp.posToXY(pos, count)
	drawHexagon(da, cr, dp.face, x, y)
}

// drawHexagonSelection draws the hexagon with the colors for piece selection.
func drawHexagonSelection(da *gtk.DrawingArea, cr *cairo.Context, face, xc, yc float64) {
	cr.Save()
	defer cr.Restore()

	cr.SetLineWidth(5.0)
	cr.SetLineJoin(cairo.LINE_JOIN_ROUND)
	cr.SetSourceRGB(0.38, 0.114, 0.549)
	drawHexagon(da, cr, standardFace, xc, yc)
}

func drawPieceAndBase(da *gtk.DrawingArea, cr *cairo.Context, player uint8, piece PieceType, face, xc, yc float64) {
	drawPieceBase(da, cr, player, face, xc, yc)
	drawPiece(da, cr, piece, face, xc, yc)
}

func drawPiece(da *gtk.DrawingArea, cr *cairo.Context, piece PieceType, face, xc, yc float64) {
	drawPieceSurface(da, cr, pieceSurfaces[piece], face*0.75, xc, yc)
}

func drawPieceBase(da *gtk.DrawingArea, cr *cairo.Context, player uint8, face, xc, yc float64) {
	drawPieceSurface(da, cr, pieceBaseSurfaces[player], face*1.15, xc, yc)
}

func drawPieceSurface(da *gtk.DrawingArea, cr *cairo.Context, surface *cairo.Surface, face, xc, yc float64) {
	cr.Save()
	defer cr.Restore()

	imgWidth, imgHeight := float64(surface.GetWidth()), float64(surface.GetHeight())
	tgtWidth, tgtHeigth := 2.*face, 2.*hexTriangleHeight(face)

	// Scale image such that it fits within 2xface width, and 2xface height (but don't up-scale).
	sx, sy := tgtWidth/imgWidth, tgtHeigth/imgHeight
	if sx > 1.0 {
		sx = 1.0
	}
	if sy < sx {
		sx = sy
	} else {
		sy = sx
	}
	adjImgWidth, adjImgHeight := imgWidth*sx, imgHeight*sy
	cr.Scale(sx, sy)

	// Find x/y on the rescaled dimensions.
	tgtX, tgtY := xc-adjImgWidth/2, yc-adjImgHeight/2
	tgtX /= sx
	tgtY /= sy

	// Finally paint it.
	cr.SetSourceSurface(surface, tgtX, tgtY)
	cr.Paint()
	cr.Clip()
}

func drawFullSurface(da *gtk.DrawingArea, cr *cairo.Context, surface *cairo.Surface) {
	cr.Save()
	defer cr.Restore()

	allocation := da.GetAllocation()
	width, height := float64(allocation.GetWidth()), float64(allocation.GetHeight())
	imgWidth, imgHeight := float64(surface.GetWidth()), float64(surface.GetHeight())

	// Scale image such that it fits within window (but don't up-scale).
	sx, sy := width/imgWidth, height/imgHeight
	if sx > 1.0 {
		sx = 1.0
	}
	if sy < sx {
		sx = sy
	} else {
		sy = sx
	}
	adjImgWidth, adjImgHeight := imgWidth*sx, imgHeight*sy

	// Paint image at center.
	cr.Scale(sx, sy)
	cr.SetSourceSurface(pieceSurfaces[3], (width-adjImgWidth)/2.0/sx, (height-adjImgHeight)/2.0/sy)
	cr.Paint()
}
