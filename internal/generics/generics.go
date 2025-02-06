// Package generics implements generic data structure functions missing from the stdlib.
package generics

import (
	"cmp"
	"iter"
	"maps"
	"slices"
)

// SliceMap executes the given function sequentially for every element on in, and returns a mapped slice.
func SliceMap[In, Out any](in []In, fn func(e In) Out) (out []Out) {
	out = make([]Out, len(in))
	for ii, e := range in {
		out[ii] = fn(e)
	}
	return
}

// SortedKeys returns an iterator over the sorted keys of the given map.
//
// It extracts the keys, sort them and then iterate over, so it's convenient but not fast.
func SortedKeys[M interface{ ~map[K]V }, K cmp.Ordered, V any](m M) iter.Seq[K] {
	sortedKeys := slices.Collect(maps.Keys(m))
	slices.Sort(sortedKeys)
	return slices.Values(sortedKeys)
}

// SortedKeysAndValues returns an interator over keys and values of a map m in a sorted fashion by the keys.
//
// It extracts the keys, sort them and then iterate over, so it's convenient but not fast.
func SortedKeysAndValues[M interface{ ~map[K]V }, K cmp.Ordered, V any](m M) iter.Seq2[K, V] {
	sortedKeys := slices.Collect(maps.Keys(m))
	slices.Sort(sortedKeys)
	return func(yield func(K, V) bool) {
		for _, key := range sortedKeys {
			if !yield(key, m[key]) {
				break
			}
		}
	}
}

// Pair defines a pair of 2 different arbitrary pairs.
type Pair[A, B any] struct {
	A A
	B B
}

// CollectPairs from an interator with 2 values.
func CollectPairs[A, B any](seq iter.Seq2[A, B]) []Pair[A, B] {
	var pairs []Pair[A, B]
	for a, b := range seq {
		pairs = append(pairs, Pair[A, B]{a, b})
	}
	return pairs
}

// Set implements a Set for the key type T.
type Set[T comparable] map[T]struct{}

// MakeSet returns an empty Set of the given type. Size is optional, and if given
// will reserve the expected size.
func MakeSet[T comparable](size ...int) Set[T] {
	if len(size) == 0 {
		return make(Set[T])
	}
	return make(Set[T], size[0])
}

// SetWith creates a Set[T] with the given elements inserted.
func SetWith[T comparable](elements ...T) Set[T] {
	s := MakeSet[T](len(elements))
	for _, element := range elements {
		s.Insert(element)
	}
	return s
}

// Has returns true if Set s has the given key.
func (s Set[T]) Has(key T) bool {
	_, found := s[key]
	return found
}

// Insert keys into set.
func (s Set[T]) Insert(keys ...T) {
	for _, key := range keys {
		s[key] = struct{}{}
	}
}

// Sub returns `s - s2`, that is, all elements in `s` that are not in `s2`.
func (s Set[T]) Sub(s2 Set[T]) Set[T] {
	sub := MakeSet[T]()
	for k := range s {
		if !s2.Has(k) {
			sub.Insert(k)
		}
	}
	return sub
}
