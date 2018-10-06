// Package state holds information about a game state.
//
// This file holds the definition of Surface of a hive. These are connected
// areas where an ANT can move. Usually there is only an external one,
// sorrounding the hive. But there can more internal ones.
package state

// Surface is a connected area of the hive space, where an ANT
// could presumably move.
type Surface map[Pos]bool

// Surfaces represent the collection of Surface objects that represent the areas where
// an ANT can move while connected to a HIVE.
type Surfaces []Surface
