package generics

import (
	"github.com/stretchr/testify/assert"
	"slices"
	"testing"
)

func TestSortedKeys(t *testing.T) {
	m := map[int]string{1: "1", 5: "5", 3: "3"}
	// Since the builtin map iterator in Go is deliberately non-deterministic, we
	// run it a bunch of times to show it is stably sorted.
	want := []int{1, 3, 5}
	for _ = range 100 {
		got := slices.Collect(SortedKeys(m))
		if !slices.Equal(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
	}
}

func TestSortedKeysAndValues(t *testing.T) {
	m := map[int]string{1: "1", 5: "5", 3: "3"}
	// Since the builtin map iterator in Go is deliberately non-deterministic, we
	// run it a bunch of times to show it is stably sorted.
	want := []Pair[int, string]{{1, "1"}, {3, "3"}, {5, "5"}}
	for _ = range 100 {
		got := CollectPairs(SortedKeysAndValues(m))
		if !slices.Equal(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
	}
}

func TestSet(t *testing.T) {
	// Sets are created empty.
	s := MakeSet[int](10)
	assert.Len(t, s, 0)

	// Check inserting and recovery.
	s.Insert(3, 7)
	assert.Len(t, s, 2)
	assert.True(t, s.Has(3))
	assert.True(t, s.Has(7))
	assert.False(t, s.Has(5))

	s2 := SetWith(5, 7)
	assert.Len(t, s2, 2)
	assert.True(t, s2.Has(5))
	assert.True(t, s2.Has(7))
	assert.False(t, s2.Has(3))

	s3 := s.Sub(s2)
	assert.Len(t, s3, 1)
	assert.True(t, s3.Has(3))

	delete(s, 7)
	assert.Len(t, s, 1)
	assert.True(t, s.Has(3))
	assert.False(t, s.Has(7))
	assert.True(t, s.Equal(s3))
	assert.False(t, s.Equal(s2))
	s4 := SetWith(-3)
	assert.False(t, s.Equal(s4))
}

func TestSliceOrdering(t *testing.T) {
	s := []float32{7, -3, 2}
	assert.Equal(t, []int{1, 2, 0}, SliceOrdering(s, false))
	s2 := []int64{0, 1, 2}
	assert.Equal(t, []int{2, 1, 0}, SliceOrdering(s2, true))
}
