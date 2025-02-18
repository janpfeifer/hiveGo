package main

import (
	. "github.com/janpfeifer/hiveGo/internal/state"
)

// This file implements the training once one has the training data (LabeledBoards).
//
// It also defines LabeledBoards, the container of with the boards and labels to train.

// LabeledBoards is a container of board positions and its labels (scores), used for training.
//
// ActionsLabels are optional. If set, they must match the number of actions of the corresponding board.
type LabeledBoards struct {
	Boards        []*Board
	Labels        []float32
	ActionsLabels [][]float32

	// MaxSize configures the max number of boards to hold in LabeledBoards, if > 0.
	// After it reaches the MaxSize new boards appended start rotating the position (replacing older ones).
	MaxSize, CurrentIdx int
}

// Len returns the number of board positions stored.
func (lb *LabeledBoards) Len() int {
	return len(lb.Boards)
}

// AddBoard and its labels to LabeledBoards collection.
// If LabeledBoards has a MaxSize configured, and it is full, it starts recycling its buffer to
// give space to new board.
func (lb *LabeledBoards) AddBoard(board *Board, label float32, actionsLabels []float32) {
	if lb.MaxSize == 0 || lb.Len() < lb.MaxSize {
		// AddBoard to the end.
		lb.Boards = append(lb.Boards, board)
		lb.Labels = append(lb.Labels, label)
		if actionsLabels != nil {
			lb.ActionsLabels = append(lb.ActionsLabels, actionsLabels)
		}
	} else {
		// Start cycling current buffer.
		lb.CurrentIdx = lb.CurrentIdx % lb.MaxSize
		lb.Boards[lb.CurrentIdx] = board
		lb.Labels[lb.CurrentIdx] = label
		if actionsLabels != nil {
			lb.ActionsLabels[lb.CurrentIdx] = actionsLabels
		}
	}
	lb.CurrentIdx++
}
