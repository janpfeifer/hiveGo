#!/bin/bash
LOGS_FILE="$1" ; shift
WINS_PNG="$1" ; shift
LOSS_PNG="$1" ; shift

function wins_plot() {
	WINS_DATA="$(mktemp /tmp/wins_stats.XXXXXXXXX)"
	printf '"Player 0 Wins" "Player 1 Wins" "Draws"\n' > "${WINS_DATA}"
	cat "${LOGS_FILE}" \
	| egrep  '(draw=|Loss|checkpointing)' \
	| perl -ne '/^.*p0 win=(\d+), p1 win=(\d+), draw=(\d+)/ && $1 + $2 + $3 == 100 && print "$1 $2 $3\n";' \
	>> "${WINS_DATA}"

	gnuplot <<EOF
		set term png giant size 2048,1024 font '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
		set output '${WINS_PNG}'
		plot for [i=1:3] "${WINS_DATA}" using i title columnhead(i) with lines
EOF
	rm $WINS_DATA
}

function loss_plot() {
	LOSS_DATA="$(mktemp /tmp/loss_stats.XXXXXXXXX)"
	printf '"Total Loss" "Board Loss" "Actions Loss * 1e3"\n' > ${LOSS_DATA}
	cat "${LOGS_FILE}" \
	| egrep  '(draw=|Loss|checkpointing)' \
	| tail -n +200 \
	| perl -e '$ma=0.99 ; $t_m=0.0; $b_m=0.0; $a_m=0.0; $c=0;
	 	while (<>) { 
	 		if ( m/.*Loss after epoch: total=(.*?), board=(.*?), actions=(.*)$/ ) { 
	 			$t=$1; $b=$2; $a=$3 ; 
	 			$r=$c/($c+1) ; if ($r>$ma) { $r=$ma } ; $r1=1.0-$r ; 
	 			$t_m=$t_m*$r+$t*$r1; 
	 			$b_m=$b_m*$r+$b*$r1; 
	 			$a_m=$a_m*$r+$a*$r1; 
	 			$aa = 1000.0*$a_m ;
	 			$c++ ; 
	 			print "$t_m $b_m $aa\n"; 
	 		} 
	 	}' \
	>> "${LOSS_DATA}"

	gnuplot <<EOF
		set term png giant size 2048,1024 font '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
		set output '${LOSS_PNG}'
		plot for [i=1:3] "${LOSS_DATA}" using i title columnhead(i) with lines
EOF
	rm $LOSS_DATA
}


wins_plot
loss_plot