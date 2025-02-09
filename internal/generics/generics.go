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

// KeysSlice returns a slice with the keys of a map.
func KeysSlice[Map interface{ ~map[K]V }, K comparable, V any](m Map) []K {
	keys := make([]K, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// SortedKeys returns an iterator over the sorted keys of the given map.
//
// It extracts the keys, sort them and then iterate over, so it's convenient but not fast.
func SortedKeys[M interface{ ~map[K]V }, K cmp.Ordered, V any](m M) iter.Seq[K] {
	sortedKeys := KeysSlice(m)
	slices.Sort(sortedKeys)
	return slices.Values(sortedKeys)
}

// SortedKeysAndValues returns an interator over keys and values of a map m in a sorted fashion by the keys.
//
// It extracts the keys, sort them and then iterate over, so it's convenient but not fast.
func SortedKeysAndValues[Map interface{ ~map[K]V }, K cmp.Ordered, V any](m Map) iter.Seq2[K, V] {
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

// MapAnyKey returns any one key from the map. Non-deterministic, each time a different one could be returned.
//
// This is akin to accessing slice[0], it will panic if the map is empty.
func MapAnyKey[Map interface{ ~map[K]V }, K comparable, V any](m Map) K {
	for k, _ := range m {
		return k
	}
	panic("map is empty, no key exists")
}

// MapAnyValue returns any one value from the map. Non-deterministic, each time a different one could be returned.
//
// This is akin to accessing slice[0], it will panic if the map is empty.
func MapAnyValue[Map interface{ ~map[K]V }, K comparable, V any](m Map) V {
	for _, v := range m {
		return v
	}
	panic("map is empty, no value exists")
}

// Pair defines a pair of 2 different arbitrary pairs.
type Pair[F, S any] struct {
	First  F
	Second S
}

// CollectPairs from an interator with 2 values.
func CollectPairs[F, S any](seq iter.Seq2[F, S]) []Pair[F, S] {
	var pairs []Pair[F, S]
	for a, b := range seq {
		pairs = append(pairs, Pair[F, S]{a, b})
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

// Equal returns whether s and s2 have the exact same elements.
func (s Set[T]) Equal(s2 Set[T]) bool {
	if len(s) != len(s2) {
		return false
	}
	for k := range s {
		if !s2.Has(k) {
			return false
		}
	}
	return true
}

// SliceOrdering return a slice of indices to s (the original slice) that points
// to them in order -- without any changes to s.
//
// If reverse is set to true, it returns the reverse order instead.
func SliceOrdering[S interface{ ~[]E }, E cmp.Ordered](s S, reverse bool) []int {
	ordering := make([]int, len(s))
	for i := range ordering {
		ordering[i] = i
	}
	reverseMult := 1
	if reverse {
		reverseMult = -1
	}
	slices.SortFunc(ordering, func(a, b int) int {
		result := cmp.Compare(s[a], s[b])
		return result * reverseMult
	})
	return ordering
}
