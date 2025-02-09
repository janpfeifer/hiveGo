// Package cli implements a command-line UI for the game.
package cli

import (
	"bufio"
	"errors"
	"fmt"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"log"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	LinesPerRow    = 4
	CharsPerColumn = 9
)

func centerString(s string, fit int) string {
	if len(s) >= fit {
		return s
	}
	marginLeft := (fit - len(s)) / 2
	marginRight := fit - len(s) - marginLeft
	return strings.Repeat(" ", marginLeft) + s + strings.Repeat(" ", marginRight)
}

type UI struct {
	board              *Board
	color, clearScreen bool
	reader             *bufio.Reader
}

var (
	placementParser = regexp.MustCompile(`^\s*(\w)[\s,]+(-?\d+)[\s,]+(-?\d+)[\s,]*$`)
	moveParser      = regexp.MustCompile(`^\s*(-?\d+)[\s,]+(-?\d+)[\s,]+(-?\d+)[\s,]+(-?\d+)[\s,]*$`)

	parsingErrorMsg = "failed to read command 3 times"
)

func New(color bool, clearScreen bool) *UI {
	return &UI{
		color:       color,
		clearScreen: clearScreen,
		reader:      bufio.NewReader(os.Stdin),
	}
}

func (ui *UI) CheckNoAvailableAction(board *Board) (*Board, bool) {
	if len(board.Derived.Actions) > 0 {
		return board, false
	}

	// Nothing to play, skip (by playing SkipAction)
	fmt.Println()
	ui.PrintPlayer(board)
	fmt.Println(" has no available actions, skipping.")
	fmt.Println()
	board = board.Act(SkipAction)
	return board, true
}

func (ui *UI) RunNextMove(board *Board) (*Board, error) {
	for {
		ui.Print(board)
		action, err := ui.ReadCommand(board)
		if err != nil && err.Error() == parsingErrorMsg {
			continue
		}
		if err != nil {
			log.Printf("Run() failed: %s", err)
			return board, err
		}
		board = board.Act(action)
		break
	}
	return board, nil
}

func (ui *UI) Run(board *Board) (*Board, error) {
	for {
		board, _ = ui.CheckNoAvailableAction(board)
		if board.IsFinished() {
			ui.PrintWinner(board)
			return board, nil
		}
		var err error
		board, err = ui.RunNextMove(board)
		if err != nil {
			return board, err
		}
	}
}

func (ui *UI) PrintWinner(b *Board) {
	d := b.Derived
	if d.Wins[0] && d.Wins[1] {
		reason := "Both queens were surrounded"
		if d.Repeats >= 2 {
			reason = "Last position repeated 3 times"
		} else if b.MoveNumber > b.MaxMoves {
			reason = "Max number of moves reached"
		}
		fmt.Printf("\n\n%s*** DRAW: %s! ***%s\n\n",
			ui.blinkStart(), reason, ui.colorEnd())
	} else {
		player := PlayerFirst
		if d.Wins[1] {
			player = PlayerSecond
		}
		fmt.Printf("\n\n%s*** %s PLAYER WINS!! Congratulations! ***%s\n\n",
			ui.colorStart(player, QUEEN), strings.ToUpper(player.String()), ui.colorEnd())
	}
}

func (ui *UI) ReadCommand(b *Board) (action Action, err error) {
	for numErrs := 0; numErrs < 3; numErrs++ {
		fmt.Println()
		ui.PrintPlayer(b)
		fmt.Print(": ")
		var text string
		text, err = ui.reader.ReadString('\n')
		if err != nil {
			return
		}
		matches := placementParser.FindStringSubmatch(strings.ToUpper(text))
		if len(matches) == 4 {
			// Placement action.
			action.Move = false
			action.SourcePos = Pos{0, 0}
			if piece, ok := LetterToPiece[matches[1]]; !ok {
				fmt.Printf("Sorry insect '%s' unknown, choose one of 'A', 'B', 'G', 'Q', 'S'\n",
					matches[1])
				continue
			} else {
				action.Piece = piece
			}
			failed := false
			for ii := 0; ii < 2; ii++ {
				if i64, err := strconv.ParseInt(matches[2+ii], 10, 8); err != nil {
					fmt.Printf("Failed to parse location '%s' in '%s'\n", matches[2+ii], text)
					failed = true
					break
				} else {
					action.TargetPos[ii] = int8(i64)
				}
			}
			if failed {
				continue
			}
			if !b.IsValid(action) {
				fmt.Printf("Placing %s in %s is not a valid placement.\n",
					action.Piece, action.TargetPos)
				continue
			}
			// Correctly parsed placement move.
			err = nil
			return
		}
		matches = moveParser.FindStringSubmatch(strings.ToUpper(text))
		if len(matches) == 5 {
			// Move action.
			action.Move = true
			failed := false
			for ii := 0; ii < 4; ii++ {
				if i64, err := strconv.ParseInt(matches[1+ii], 10, 8); err != nil {
					fmt.Printf("Failed to parse location '%s' in '%s'\n", matches[1+ii], text)
					failed = true
					break
				} else {
					if ii < 2 {
						action.SourcePos[ii] = int8(i64)
					} else {
						action.TargetPos[ii-2] = int8(i64)
					}
				}
			}
			if failed {
				continue
			}
			_, action.Piece, _ = b.PieceAt(action.SourcePos)
			if !b.IsValid(action) {
				fmt.Printf("Moving %s from %s to %s is not valid.\n",
					action.Piece, action.SourcePos, action.TargetPos)
				if b.Available(b.NextPlayer, QUEEN) != 0 {
					fmt.Printf("One can only start moving pieces once the Queen is on the board.\n")
				}
				continue
			}
			err = nil
			return
		}
		fmt.Printf("Failed to parse your input '%s', please try again.", text)
	}
	err = errors.New(parsingErrorMsg)
	return
}

func (ui *UI) Print(board *Board) {
	if board.Derived == nil {
		log.Fatal("Called UI.Print(board), with board without Derived set.")
	}
	if ui.clearScreen {
		fmt.Print("\033c")
		// fmt.Print("\033[2J")
	}
	if ui.color {
		fmt.Print("\033[37;03;1m")
	}
	fmt.Printf("\nMove #%d%s\n\n", board.MoveNumber, ui.colorEnd())

	ui.PrintBoard(board)
	fmt.Println()
	ui.PrintAvailable(board)
	fmt.Print("\n")
	ui.PrintPlayer(board)
	fmt.Print(" turn to play\n")
	ui.printActions(board)
	fmt.Println()
}

func (ui *UI) PrintPlayer(board *Board) {
	fmt.Printf("%sPlayer %d%s", ui.colorStart(board.NextPlayer, QUEEN), board.NextPlayer, ui.colorEnd())
}

func (ui *UI) PrintAvailable(board *Board) {
	for _, player := range []PlayerNum{PlayerFirst, PlayerSecond} {
		var pieces []string
		for _, piece := range Pieces {
			pieces = append(pieces, fmt.Sprintf("%s-%d", PieceLetters[piece],
				board.Available(player, piece)))
		}
		sort.Strings(pieces)
		fmt.Printf("Player %d available: [%s]\n", player, strings.Join(pieces, ", "))
	}
}

func (ui *UI) PrintBoard(board *Board) {
	minX, maxX, minY, maxY := board.DisplayUsedLimits()
	minX--
	maxX++
	minY--
	maxY++
	// Loop over board rows.
	for y := minY; y <= maxY; y++ {
		// Loop over line within a row.
		for line := int8(0); line < LinesPerRow; line++ {
			ui.printBoardLine(board, y, line, minX, maxX)
		}
	}
}

func (ui *UI) printBoardLine(board *Board, y, line, minX, maxX int8) {
	for x := minX; x <= maxX+1; x++ {
		adjY := y
		adjLine := line
		if x%2 != 0 {
			adjLine = (line - LinesPerRow/2 + LinesPerRow) % LinesPerRow
			if adjLine >= 2 {
				adjY -= 1
			}
		}
		// The display coordinate (the X,Y in the screen) doesn't exactly map to the
		// state position (the X,Y of the hexagonal positions).
		pos := Pos{x, adjY}.FromDisplayPos()
		player, piece, stacked := board.PieceAt(pos)
		lastX := x == maxX+1
		ui.printStrip(board, pos, player, piece, stacked, adjLine, lastX)
	}
	fmt.Println()
}

func (ui *UI) printStrip(board *Board, pos Pos,
	player PlayerNum, piece PieceType, stacked bool, line int8, lastX bool) {
	switch {
	case line == 0:
		fmt.Print(" /")
		if !lastX {
			fmt.Print(strings.Repeat(" ", CharsPerColumn-2))
		}
	case line == 1:
		fmt.Print("/")
		if !lastX {
			coord := fmt.Sprintf("%d,%d", pos.X(), pos.Y())
			fmt.Print(" " + centerString(coord, CharsPerColumn-2))
		}
	case line == 2:
		fmt.Print("\\")
		if !lastX {
			if piece == NoPiece {
				fmt.Print(strings.Repeat(" ", CharsPerColumn-1))
			} else {
				fmt.Print(" ")
				if !stacked {
					fmt.Print(ui.colorStart(player, piece) +
						centerString(PieceLetters[piece], CharsPerColumn-2) +
						ui.colorEnd())
				} else {
					stack := board.StackAt(pos)
					fmt.Print(ui.stackedPieces(player, piece, stack, CharsPerColumn-2))
				}
			}
		}
	case line == 3:
		fmt.Print(" \\")
		if !lastX {
			fmt.Print(strings.Repeat("_", CharsPerColumn-2))
		}
	}
}

// colorStart returns the string to start a color appropriate for the given
// player/piece pair.
func (ui *UI) colorStart(player PlayerNum, piece PieceType) string {
	if !ui.color {
		return ""
	}
	if player == PlayerFirst {
		if piece == QUEEN {
			return "\033[37;41;1m"
		} else {
			return "\033[30;41;1m"
		}
	} else {
		if piece == QUEEN {
			return "\033[37;42;1m"
		} else {
			return "\033[30;42;1m"
		}
	}
}

func (ui *UI) colorEnd() string {
	if !ui.color {
		return ""
	}
	return "\033[39;49;0m"
}

func (ui *UI) blinkStart() string {
	if !ui.color {
		return ""
	}
	return "\033[5m"
}

func (ui *UI) stackedPieces(player PlayerNum, piece PieceType, stack EncodedStack, fit int) string {
	numPieces := int(stack.CountPieces())
	totalLen := 2 + numPieces
	marginLeft := (fit - totalLen) / 2
	if marginLeft < 0 {
		marginLeft = 0
	}
	marginRight := fit - totalLen - marginLeft
	if marginRight < 0 {
		marginRight = 0
	}

	str := ui.colorStart(player, piece)
	str += strings.Repeat(" ", marginLeft)
	str += PieceLetters[piece]
	str += "("
	str += ui.colorEnd()
	for i := 1; i < numPieces; i++ {
		stackedPlayer, stackedPiece := stack.PieceAt(uint8(i))
		str += ui.colorStart(stackedPlayer, stackedPiece)
		str += PieceLetters[stackedPiece]
		str += ui.colorEnd()
	}
	str += ui.colorStart(player, piece)
	str += ")" + strings.Repeat(" ", marginRight)
	str += ui.colorEnd()
	return str
}

func (ui *UI) printActions(b *Board) {
	fmt.Print("Available actions:\n")
	ui.printPlacementActions(b)
	fmt.Println()
	ui.printMoveActions(b)
}

func (ui *UI) printPlacementActions(b *Board) {
	d := b.Derived
	if len(d.PlacementPositions[b.NextPlayer]) == 0 {
		return
	}

	// Set of pieces that can be placed.
	pieces := make(map[PieceType]bool)
	for _, action := range d.Actions {
		if !action.Move {
			pieces[action.Piece] = true
		}
	}
	if len(pieces) == 0 {
		return
	}
	piecesStr := make([]string, 0, len(Pieces))
	var examplePiece PieceType
	for p := range pieces {
		piecesStr = append(piecesStr, PieceLetters[p])
		if examplePiece == NoPiece {
			examplePiece = p
		}
	}

	// List placement positions.
	positionsMap := make(map[Pos]bool)
	for _, action := range d.Actions {
		if !action.Move {
			positionsMap[action.TargetPos] = true
		}
	}
	positions := make([]Pos, 0, len(positionsMap))
	for pos := range positionsMap {
		positions = append(positions, pos)
	}
	PosSort(positions)
	examplePos := positions[0]

	// Print the available pieces / positions, and an example.
	fmt.Printf("  * place piece [%s] in one of the positions [%s]\n",
		strings.Join(piecesStr, ", "), strings.Join(PosStrings(positions), ", "))
	fmt.Printf("    Example: type '%s %d %d' to place a %s in %s\n",
		piecesStr[0], examplePos.X(), examplePos.Y(), examplePiece, examplePos)
}

func (ui *UI) printMoveActions(b *Board) {
	d := b.Derived

	// List pieces that can be placed.
	pieces := make(map[Pos][]Action)
	for _, action := range d.Actions {
		if action.Move {
			actions, _ := pieces[action.SourcePos]
			pieces[action.SourcePos] = append(actions, action)
		}
	}
	if len(pieces) == 0 {
		return
	}

	// Sort source positions.
	srcPositions := make([]Pos, 0, len(pieces))
	for srcPos := range pieces {
		srcPositions = append(srcPositions, srcPos)
	}
	PosSort(srcPositions)

	// Print actions organized by source position.
	var (
		examplePiece                 PieceType
		exampleSrcPos, exampleTgtPos Pos
	)
	for ii, srcPos := range srcPositions {
		piece := pieces[srcPos][0].Piece
		tgtPoss := make([]Pos, 0, len(pieces[srcPos]))
		for _, action := range pieces[srcPos] {
			tgtPoss = append(tgtPoss, action.TargetPos)
		}
		PosSort(tgtPoss)

		if ii == 0 {
			examplePiece = piece
			exampleSrcPos = srcPos
			exampleTgtPos = tgtPoss[0]
		}
		fmt.Printf("  * Move %s at %s to one of the positions [%s]\n",
			PieceLetters[piece], srcPos, strings.Join(PosStrings(tgtPoss), ", "))
	}
	fmt.Printf("    Example: to move %s at %s to %s, type the source and target positions: '%d %d %d %d'",
		PieceNames[examplePiece], exampleSrcPos, exampleTgtPos,
		exampleSrcPos.X(), exampleSrcPos.Y(),
		exampleTgtPos.X(), exampleTgtPos.Y())
}
