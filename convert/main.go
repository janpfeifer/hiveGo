package main

import (
	"bufio"
	"encoding/gob"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"strconv"

	"github.com/golang/glog"
	. "github.com/janpfeifer/hiveGo/state"
)

var (
	flag_input  = flag.String("input", "", "Text file with moves.")
	flag_output = flag.String("output", "", "New saved file.")
	flag_wins   = flag.Bool("wins", false, "Converts only matches with a win.")


	reMove = regexp.MustCompile(`Move ([ABGQS]): \((-?\d+), (-?\d+)\)->\((-?\d+), (-?\d+)\)`)
	rePlace = regexp.MustCompile(`Place ([ABGQS]) in \((-?\d+), (-?\d+)\)`)
)

type Match struct {
	MaxMoves int
	Actions []Action
	Scores []float32
}

func loadMatches(results chan<- *Match) {
	file, err := os.Open(*flag_input)
	if err != nil {
		log.Panicf("Cannot open '%s' for reading: %v", *flag_output, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	board := NewBoard()
	match := &Match{MaxMoves:board.MaxMoves}
	for scanner.Scan() {
		line := scanner.Text()
		var action Action
		if parts := reMove.FindStringSubmatch(line); parts != nil {
			action.Move = true
			action.Piece = LetterToPiece[parts[1]]
			x, _ := strconv.Atoi(parts[2])
			y, _ := strconv.Atoi(parts[3])
			action.SourcePos = Pos{int8(x), int8(y)}
			x, _ = strconv.Atoi(parts[4])
			y, _ = strconv.Atoi(parts[5])
			action.TargetPos = Pos{int8(x), int8(y)}
		} else if parts = rePlace.FindStringSubmatch(line); parts != nil {
			action.Move = false
			action.Piece = LetterToPiece[parts[1]]
			x, _ := strconv.Atoi(parts[2])
			y, _ := strconv.Atoi(parts[3])
			action.TargetPos = Pos{int8(x), int8(y)}
		} else {
			glog.Errorf("Unparsed line: [%s]", line)
			continue
		}
		_ = board.FindActionDeep(action)
		board = board.Act(action)
		match.Actions = append(match.Actions, action)
		match.Scores = append(match.Scores, 0)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	if !board.IsFinished() {
		glog.Errorf("Match hasn't finished at move number %d", board.MoveNumber)
	}
	glog.Infof("Converted match with %d moves", board.MoveNumber)
	results <- match
	close(results)
}

func openWriterAndBackup(filename string) io.WriteCloser {
	if _, err := os.Stat(filename); err == nil {
		err = os.Rename(filename, filename+"~")
		if err != nil {
			log.Printf("Failed to rename '%s' to '%s~': %v", filename, filename, err)
		}
	}
	file, err := os.Create(filename)
	if err != nil {
		log.Panicf("Failed to save file to '%s': %v", filename, err)
	}
	return file
}

func main() {
	flag.Parse()
	results := make(chan *Match)
	go loadMatches(results)

	file := openWriterAndBackup(*flag_output)
	enc := gob.NewEncoder(file)
	count := 0
	for match := range results {
		SaveMatch(enc, match.MaxMoves, match.Actions, match.Scores)
		count++
	}
	file.Close()
	fmt.Printf("%d matches converted.\n", count)
}
