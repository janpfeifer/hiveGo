// Package cli implements a command-line UI for the game.
package cli

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"github.com/charmbracelet/lipgloss"
	. "github.com/janpfeifer/hiveGo/internal/state"
	"golang.org/x/term"
	"io"
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

var ansiFilter = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)

// displayWidth of s removes its color/control sequences and returns the length of what is left.
func displayWidth(s string) int {
	return len(ansiFilter.ReplaceAllString(s, ""))
}

func printCentered(block string) {
	lines := strings.Split(block, "\n")
	terminalWidth, _, _ := term.GetSize(int(os.Stdout.Fd()))
	blockWidth := 0
	for _, line := range lines {
		if len(line) > blockWidth {
			blockWidth = displayWidth(line)
		}
	}
	indent := (terminalWidth - blockWidth) / 2
	if indent < 0 {
		indent = 0
	}
	for _, line := range lines {
		if len(line) == 0 {
			fmt.Println()
			continue
		}
		fmt.Printf("%s%s\n", strings.Repeat(" ", indent), line)
	}
}

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
	if len(board.Derived.Actions) > 1 || board.Derived.Actions[0] != SkipAction {
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
		ui.Print(board, true)
		fmt.Println()
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
	winner := b.Winner()
	fmt.Println()
	if winner == PlayerInvalid {
		printCentered(
			lipgloss.NewStyle().
				Background(lipgloss.Color("13")).
				Foreground(lipgloss.Color("0")).
				Padding(1, 2).
				Render(fmt.Sprintf("*** DRAW: %s! ***", b.FinishReason())))
	} else {
		printCentered(fmt.Sprintf("%s *** %s PLAYER WINS!! Congratulations! *** %s\n",
			ui.colorStart(winner), strings.ToUpper(winner.String()), ui.colorEnd()))
	}
	fmt.Println()
}

func (ui *UI) ReadCommand(b *Board) (action Action, err error) {
	// ANSI escape codes for:
	// - \033[7m:  Reverse video (swap foreground and background)
	// - \033[45m: Set background color to magenta (purple-ish)
	// - \033[0m:  Reset all attributes to defaults
	const (
		inputAreaColor = "\033[30;45;2m"        // Purplish background
		inputAreaReset = "\033[39;49;0m\033[0K" // Reset color and clear to the end-of-line.
		inputWidth     = 14                     // Width of the input area
	)

	for numErrs := 0; numErrs < 3; numErrs++ {
		fmt.Print("    ")
		ui.PrintPlayer(b)
		fmt.Print(" action > ")

		// Print "input area" in purple, and move the cursor back to the beginning of the input area.
		fmt.Printf("%s%s", inputAreaColor, strings.Repeat(" ", inputWidth))
		fmt.Printf("\033[%dD", inputWidth-1) // Left 1 char padding.

		var text string
		text, err = ui.reader.ReadString('\n')
		fmt.Printf(inputAreaReset) // We don't want the purple color to leak.
		if err != nil {
			return
		}
		text = strings.TrimSpace(text)

		matches := placementParser.FindStringSubmatch(strings.ToUpper(text))
		if len(matches) == 4 {
			// Placement action.
			action.Move = false
			action.SourcePos = Pos{0, 0}
			if piece, ok := LetterToPiece[matches[1]]; !ok {
				fmt.Printf("    * Sorry insect %q unknown, choose one of 'A', 'B', 'G', 'Q', 'S'\n",
					matches[1])
				continue
			} else {
				action.Piece = piece
			}
			failed := false
			for ii := 0; ii < 2; ii++ {
				if i64, err := strconv.ParseInt(matches[2+ii], 10, 8); err != nil {
					fmt.Printf("    * Failed to parse location %q in %q\n", matches[2+ii], text)
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
				fmt.Printf("    * Placing %s in %s is not a valid placement.\n",
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
					fmt.Printf("    * Failed to parse location %q in %q\n", matches[1+ii], text)
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
		fmt.Printf("    * Failed to parse your input %q, please try again.\n", text)
	}
	err = errors.New(parsingErrorMsg)
	return
}

func (ui *UI) Print(board *Board, includeAvailableActions bool) {
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
	ui.PrintAvailablePieces(board)

	if !board.IsFinished() {
		if includeAvailableActions {
			fmt.Println()
			ui.PrintPlayer(board)
			fmt.Println(" turn to play")
			ui.printActions(board)
		} else {
			fmt.Print("\tTurn to play: ")
			ui.PrintPlayer(board)
			fmt.Println()
		}
	}
}

func (ui *UI) PrintPlayer(board *Board) {
	fmt.Printf("%s%s Player%s", ui.colorStart(board.NextPlayer), board.NextPlayer, ui.colorEnd())
}

// PrintSpacedPlayer is like PrintPlayer, but includes a left-space for the first player,
// so they all use the same width.
func (ui *UI) PrintSpacedPlayer(board *Board) {
	if board.NextPlayer == PlayerFirst {
		fmt.Print(" ")
	}
	ui.PrintPlayer(board)
}

func (ui *UI) PrintAvailablePieces(board *Board) {
	for _, player := range []PlayerNum{PlayerFirst, PlayerSecond} {
		pieces := make([]string, 0, NumPieceTypes)
		for _, piece := range Pieces {
			numAvailable := board.Available(player, piece)
			if numAvailable > 0 {
				pieces = append(pieces, fmt.Sprintf("%s-%d", PieceNames[piece], numAvailable))
			}
		}
		sort.Strings(pieces)
		space := ""
		if player == 0 {
			space = " "
		}
		fmt.Printf("%s%s%s Player%s off-board: [%s]\n",
			space, ui.colorStart(player), player, ui.colorEnd(),
			strings.Join(pieces, ", "))
	}
}

func (ui *UI) PrintBoard(board *Board) {
	var buf bytes.Buffer
	minX, maxX, minY, maxY := board.DisplayUsedLimits()
	minX--
	maxX++
	minY--
	maxY++
	// Loop over board rows.
	for y := minY; y <= maxY; y++ {
		// Loop over line within a row.
		for line := int8(0); line < LinesPerRow; line++ {
			ui.printBoardLine(&buf, board, y, line, minX, maxX)
		}
	}
	printCentered(buf.String())
}

func (ui *UI) printBoardLine(w io.Writer, board *Board, y, line, minX, maxX int8) {
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
		ui.printStrip(w, board, pos, player, piece, stacked, adjLine, lastX)
	}
	_, _ = fmt.Fprintln(w)
}

func (ui *UI) printStrip(w io.Writer, board *Board, pos Pos,
	player PlayerNum, piece PieceType, stacked bool, line int8, lastX bool) {
	switch {
	case line == 0:
		_, _ = fmt.Fprint(w, " /")
		if !lastX {
			_, _ = fmt.Fprint(w, strings.Repeat(" ", CharsPerColumn-2))
		}
	case line == 1:
		_, _ = fmt.Fprint(w, "/")
		if !lastX {
			coord := fmt.Sprintf("%d,%d", pos.X(), pos.Y())
			_, _ = fmt.Fprint(w, " "+centerString(coord, CharsPerColumn-2))
		}
	case line == 2:
		_, _ = fmt.Fprint(w, "\\")
		if !lastX {
			if piece == NoPiece {
				_, _ = fmt.Fprint(w, strings.Repeat(" ", CharsPerColumn-1))
			} else {
				_, _ = fmt.Fprint(w, " ")
				if !stacked {
					_, _ = fmt.Fprint(w, ui.colorStartForPiece(player, piece)+
						centerString(PieceLetters[piece], CharsPerColumn-2)+
						ui.colorEnd())
				} else {
					stack := board.StackAt(pos)
					_, _ = fmt.Fprint(w, ui.stackedPieces(player, piece, stack, CharsPerColumn-2))
				}
			}
		}
	case line == 3:
		_, _ = fmt.Fprint(w, " \\")
		if !lastX {
			_, _ = fmt.Fprint(w, strings.Repeat("_", CharsPerColumn-2))
		}
	}
}

// colorStartForPiece returns the string to start a color appropriate for the given
// player/piece pair.
func (ui *UI) colorStartForPiece(player PlayerNum, piece PieceType) string {
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

func (ui *UI) colorStart(player PlayerNum) string {
	if !ui.color {
		return ""
	}
	if player == PlayerFirst {
		return "\033[30;41;1m"
	}
	return "\033[30;42;1m"
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

	str := ui.colorStartForPiece(player, piece)
	str += strings.Repeat(" ", marginLeft)
	str += PieceLetters[piece]
	str += "("
	str += ui.colorEnd()
	for i := 1; i < numPieces; i++ {
		stackedPlayer, stackedPiece := stack.PieceAt(uint8(i))
		str += ui.colorStartForPiece(stackedPlayer, stackedPiece)
		str += PieceLetters[stackedPiece]
		str += ui.colorEnd()
	}
	str += ui.colorStartForPiece(player, piece)
	str += ")" + strings.Repeat(" ", marginRight)
	str += ui.colorEnd()
	return str
}

func (ui *UI) printActions(b *Board) {
	fmt.Print("- Available actions:\n")
	ui.printPlacementActions(b)
	ui.printMoveActions(b)
}

func (ui *UI) printPlacementActions(b *Board) {
	d := b.Derived
	if len(d.PlacementPositions[b.NextPlayer]) == 0 {
		return
	}

	player := b.NextPlayer
	if b.Available(player, QUEEN) > 0 && b.MoveNumber >= 7 {
		fmt.Println("  - After the 3rd player's move (> 6th board move), they have to put the Queen on board")
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
	piecesStr := make([]string, 0, len(pieces))
	var examplePiece PieceType
	for p := range pieces {
		piecesStr = append(piecesStr, PieceNames[p])
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
	fmt.Printf("  - Place a piece [%s] in one of the positions [%s]\n",
		strings.Join(piecesStr, ", "), strings.Join(PosStrings(positions), ", "))
	fmt.Printf("    Example: type '%s %d %d' to place a %s in %s\n",
		PieceLetters[examplePiece], examplePos.X(), examplePos.Y(), examplePiece, examplePos)
}

func (ui *UI) printMoveActions(b *Board) {
	d := b.Derived
	player := b.NextPlayer

	// List pieces that can be placed.
	pieces := make(map[Pos][]Action)
	for _, action := range d.Actions {
		if action.Move {
			actions, _ := pieces[action.SourcePos]
			pieces[action.SourcePos] = append(actions, action)
		}
	}
	if len(pieces) == 0 {
		if b.Available(player, QUEEN) > 0 {
			fmt.Println("  - Movement of pieces not allowed until the Queen is on the board.")
		} else {
			fmt.Println("  - All your pieces are blocked, no movement is possible.")
		}
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
		fmt.Printf("  - Move %s at %s to one of the positions [%s]\n",
			PieceNames[piece], srcPos, strings.Join(PosStrings(tgtPoss), ", "))
	}
	fmt.Printf("    Example: to move %s at %s to %s, type the source and target positions: '%d %d %d %d'",
		PieceNames[examplePiece], exampleSrcPos, exampleTgtPos,
		exampleSrcPos.X(), exampleSrcPos.Y(),
		exampleTgtPos.X(), exampleTgtPos.Y())
}
