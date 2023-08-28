package state

const CACHE_SIZE = 1024 * 1024

var (
	cacheMap map[uint64][]*Board
	cacheFifo []*Board
	cacheFifoIdx = 0
	cacheFifoLen = 0
)

func init() {
	cacheMap = make(map[uint64][]*Board)
	cacheFifo = make([]*Board, CACHE_SIZE)
}

