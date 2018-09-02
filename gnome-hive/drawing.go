package main

import (
	"fmt"
	"log"

	"github.com/gotk3/gotk3/cairo"
	"github.com/gotk3/gotk3/gtk"
	. "github.com/janpfeifer/hiveGo/state"
)

var (
	pieceSurfaces     [NUM_PIECE_TYPES]*cairo.Surface
	pieceBaseSurfaces [NUM_PLAYERS]*cairo.Surface
)

// Standard face size.
const standardFace = 33.0

var (
	// Current zoom factor and translation (resulting from dragging board).
	zoomFactor     = 1.0
	shiftX, shiftY = 0.0, 0.0
)

func loadImageResources() {
	// Load each piece drawing.
	for ii := 1; ii < NUM_PIECE_TYPES; ii++ {
		surface, err := cairo.NewSurfaceFromPNG(fmt.Sprintf("%s/%s.png",
			*flag_resources, PieceNames[ii]))
		if err != nil {
			log.Fatalf("Failed to read '%s.png' from '%s': %v", PieceNames[ii], *flag_resources, err)
		}
		// log.Printf("Loaded %s.png\n", PieceNames[ii])
		pieceSurfaces[ii] = surface
	}

	// Load base piece surface for each player.
	for ii := 0; ii < NUM_PLAYERS; ii++ {
		surface, err := cairo.NewSurfaceFromPNG(fmt.Sprintf("%s/tile_player_%d.png",
			*flag_resources, ii))
		if err != nil {
			log.Fatalf("Failed to read 'tile_player_%d.png' from '%s': %v", ii, *flag_resources, err)
		}
		// log.Printf("Loaded tile_player_%d.png\n", ii)
		pieceBaseSurfaces[ii] = surface
	}
}

func drawMainBoard(da *gtk.DrawingArea, cr *cairo.Context) {
	cr.Save()
	defer cr.Restore()

	drawBackground(da, cr, 244./256., 231./256., 210./256., true)
	if !started {
		drawFullSurface(da, cr, pieceSurfaces[3])
		return
	}

	allocation := da.GetAllocation()
	width, height := float64(allocation.GetWidth()), float64(allocation.GetHeight())
	xc, yc := width/2.0+shiftX, height/2.0+shiftY
	face := standardFace * zoomFactor
	hexWidth := 1.5 * face
	hexHeight := 2. * hexTriangleHeight(face)

	poss := board.OccupiedPositions()
	PosSort(poss)
	for _, pos := range poss {
		x = xc + float64(pos.X())*hexWidth
		y = yc + float64(pos.Y())*hexHeight
		if pos.X()%2 != 0 {
			y += hexTriangleHeight(face)
		}
		player, piece, stacked := board.PieceAt(pos)
		if !stacked {
			drawPieceAndBase(da, cr, player, piece, face, x, y)
		} else {
			stack := board.StackAt(pos)
			count := int(stack.CountPieces())
			for ii := 0; ii < count; ii++ {
				player, piece = stack.PieceAt(uint8(count - ii - 1))
				drawPieceAndBase(da, cr, player, piece, face, x, y)
				x += 3.0 * zoomFactor
				y -= 3.0 * zoomFactor
			}
			// Draw small icons of pieces under the stack.
			for ii := 0; ii < count-1; ii++ {
				idx := uint8(count - ii - 1)
				player, piece = stack.PieceAt(idx)
				drawPieceAndBase(da, cr, player, piece, face/5.0,
					x+(float64(ii)-2)*0.25*face, y+0.75*hexTriangleHeight(face))
			}
		}

	}
	// const unitSize = 20.0
	// cr.SetSourceRGB(0, 0, 0)
	// cr.Rectangle(400+x*unitSize, 200+y*unitSize, unitSize, unitSize)
	// cr.Fill()
}

func drawOffBoardArea(da *gtk.DrawingArea, cr *cairo.Context, player uint8) {
	cr.Save()
	defer cr.Restore()

	allocation := da.GetAllocation()
	width, height := float64(allocation.GetWidth()), float64(allocation.GetHeight())
	xc, yc := width/2.0, height/2.0

	// Background
	drawBackground(da, cr, 0.8, 0.8, 0.8, true)
	if started && board.NextPlayer == player {
		drawBackground(da, cr, 0.6, 1.0, 0.6, false)
	}

	// Spacing between each available piece.
	const spacing = 3.0 * standardFace

	// Loop over the piece types
	for piece := 1; piece < NUM_PIECE_TYPES; piece++ {
		count := board.Available(player, Piece(piece))
		if count == 0 {
			continue
		}
		x, y := xc+(float64(piece)-3.0)*spacing, yc
		for ii := 0; ii < int(count); ii++ {
			drawPieceAndBase(da, cr, player, Piece(piece), standardFace, x+3.0*float64(ii), y-3.0*float64(ii))
		}
	}

	// Draw hexagons
	// cr.SetLineWidth(1.5)
	// cr.SetLineJoin(cairo.LINE_JOIN_ROUND)
	// cr.SetSourceRGB(0.1, 0.4, 0.1)
	// drawHexagon(da, cr, face, xc, yc)
	//
}

// hexTriangleHeight returns the height of the triangles that make up for an hexagon, given the face lenght.
func hexTriangleHeight(face float64) float64 {
	return 0.866 * face // sqrt(3)/2 * face
}

func drawBackground(da *gtk.DrawingArea, cr *cairo.Context, r, g, b float64, fill bool) {
	cr.Save()
	defer cr.Restore()

	allocation := da.GetAllocation()
	width, height := float64(allocation.GetWidth()), float64(allocation.GetHeight())
	cr.SetSourceRGB(r, g, b)
	cr.Rectangle(0.0, 0.0, width, height)
	if fill {
		cr.FillPreserve()
	} else {
		cr.SetLineWidth(3.0)
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

func drawPieceAndBase(da *gtk.DrawingArea, cr *cairo.Context, player uint8, piece Piece, face, xc, yc float64) {
	drawPieceBase(da, cr, player, face, xc, yc)
	drawPiece(da, cr, piece, face, xc, yc)
}

func drawPiece(da *gtk.DrawingArea, cr *cairo.Context, piece Piece, face, xc, yc float64) {
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
