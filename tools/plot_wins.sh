#!/bin/bash
LOGS_FILE="$1" ; shift
OUTPUT="$1" ; shift
WINS_DATA="$(mktemp /tmp/wins_stats.XXXXXXXXX)"

function createTSV() {
	printf '"Player 0 Wins" "Player 1 Wins" "Draws"\n' > "${WINS_DATA}"
	cat "${LOGS_FILE}" \
	| egrep  '(draw=|Loss|checkpointing)' /tmp/play_and_train.txt \
	| perl -ne '/^.*p0 win=(\d+), p1 win=(\d+), draw=(\d+)/ && $1 + $2 + $3 == 100 && print "$1 $2 $3\n";' \
	>> "${WINS_DATA}"
}

createTSV
gnuplot <<EOF
	set term png giant size 2048,1024 font '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
	set output '${OUTPUT}'
	plot for [i=1:3] "${WINS_DATA}" using i title columnhead(i) with lines
EOF
rm $WINS_DATA
