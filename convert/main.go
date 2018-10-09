package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/golang/glog"
	. "github.com/janpfeifer/hiveGo/state"
)

var (
	flag_input  = flag.String("input", "", "Old saved file.")
	flag_output = flag.String("output", "", "New saved file.")
	flag_wins   = flag.Bool("wins", false, "Converts only matches with a win.")
)

// Results and if the players were swapped.
type Match struct {
	// Wether p0/p1 swapped positions in this match.
	Swapped bool

	// Match actions, alternating players.
	Actions []Action

	// All board states of the game: 1 more than the number of actions.
	Boards []*Board

	// Scores for each board position. Can either be calculated during
	// the match, or re-genarated when re-loading a match.
	Scores []float32
}

func MatchDecode(dec *gob.Decoder) (match *Match, err error) {
	match = &Match{}
	err = dec.Decode(match)
	if err != nil {
		return
	}
	board := NewBoard()
	board.MaxMoves = match.Boards[0].MaxMoves
	board.BuildDerived()
	match.Boards[0] = board
	for _, action := range match.Actions {
		board = board.Act(action)
		match.Boards = append(match.Boards, board)
	}
	return
}

func (m *Match) FinalBoard() *Board { return m.Boards[len(m.Boards)-1] }

func loadMatches(results chan<- *Match) {
	file, err := os.Open(*flag_input)
	if err != nil {
		log.Panicf("Cannot open '%s' for reading: %v", *flag_output, err)
	}
	dec := gob.NewDecoder(file)

	// Run at most GOMAXPROCS re-scoring simultaneously.
	for ii := 0; true; ii++ {
		match, err := MatchDecode(dec)
		if err == io.EOF {
			break
		}
		if err != nil {
			glog.Errorf("Cannot read any more matches: %v", err)
			break
		}
		if *flag_wins && match.FinalBoard().Draw() {
			continue
		}
		results <- match
	}
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
		SaveMatch(enc, match.Boards[0].MaxMoves, match.Actions, match.Scores)
		count++
	}
	file.Close()
	fmt.Printf("%d matches converted.\n", count)
}
