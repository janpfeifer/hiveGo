package main

// BROKEN: don't use without fixing it.

import (
	"bufio"
	"encoding/gob"
	"flag"
	"fmt"
	"github.com/janpfeifer/must"
	"github.com/pkg/errors"
	"io"
	"k8s.io/klog/v2"
	"os"
	"regexp"
	"strconv"

	. "github.com/janpfeifer/hiveGo/internal/state"
)

var (
	flagInput  = flag.String("input", "", "Text file with moves.")
	flagOutput = flag.String("output", "", "New saved file.")
	flagWins   = flag.Bool("wins", false, "Converts only matches with a win.")

	reMove  = regexp.MustCompile(`Move ([ABGQS]): \((-?\d+), (-?\d+)\)->\((-?\d+), (-?\d+)\)`)
	rePlace = regexp.MustCompile(`Place ([ABGQS]) in \((-?\d+), (-?\d+)\)`)
)

type Match struct {
	MaxMoves int
	Actions  []Action
	Scores   []float32
}

// loadMatch from input file and write to given channel.
func loadMatch(results chan<- *Match) error {
	defer close(results)
	file, err := os.Open(*flagInput)
	if err != nil {

		return errors.Wrapf(err, "cannot open %q for reading", *flagOutput)
	}
	defer func() { _ = file.Close() }()

	scanner := bufio.NewScanner(file)
	board := NewBoard()
	match := &Match{MaxMoves: board.MaxMoves}
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
			klog.Errorf("Unparsed line: [%s]", line)
			continue
		}
		_ = board.FindActionDeep(action)
		board = board.Act(action)
		match.Actions = append(match.Actions, action)
		match.Scores = append(match.Scores, 0)
	}
	if err := scanner.Err(); err != nil {
		return errors.Wrapf(err, "failed to read %q", *flagInput)
	}
	if !board.IsFinished() {
		klog.Errorf("Match hasn't finished at move number %d", board.MoveNumber)
	}
	klog.Infof("Converted match with %d moves", board.MoveNumber)
	results <- match
	return nil
}

func openWriterAndBackup(filename string) (io.WriteCloser, error) {
	if _, err := os.Stat(filename); err == nil {
		backupName := filename + "~"
		err = os.Rename(filename, backupName)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to rename %q to %q", filename, backupName)
		}
	}
	file, err := os.Create(filename)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to create %q", filename)
	}
	return file, nil
}

func main() {
	flag.Parse()
	results := make(chan *Match)
	go loadMatch(results)

	file := must.M1(openWriterAndBackup(*flagOutput))
	enc := gob.NewEncoder(file)
	count := 0
	for match := range results {
		must.M(SaveMatch(enc, match.MaxMoves, match.Actions, match.Scores, nil))
		count++
	}
	must.M(file.Close())
	fmt.Printf("%d matches converted.\n", count)
}
