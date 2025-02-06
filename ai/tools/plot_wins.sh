#!/bin/bash
LOGS_FILE="$1" ; shift
WINS_PNG="$1" ; shift
LOSS_BASE="$1" ; shift
SCORES_PNG="$1" ; shift
DIFF_PNG="$1" ; shift
LOSS_BOARD_PNG="${LOSS_BASE}_board.png"
LOSS_ACTIONS_PNG="${LOSS_BASE}_actions.png"
LOSS_TOTAL_PNG="${LOSS_BASE}.png"

function wins_plot() {
	WINS_DATA="$(mktemp /tmp/wins_stats.XXXXXXXXX)"
	plot_wins < "${LOGS_FILE}" > "${WINS_DATA}"
	gnuplot <<EOF
		set term png giant size 2048,1024 font '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
		set output '${WINS_PNG}'
		plot for [i=1:3] "${WINS_DATA}" using i title columnhead(i) with lines
EOF
	rm "${WINS_DATA}"
}

function loss_plot() {
	LOSS_DATA="$(mktemp /tmp/loss_stats.XXXXXXXXX)"
	for ii in "Total Eval Loss" "Board Eval Loss" "Actions Eval Loss" \
	    "Total Train Loss" "Board Train Loss" "Actions Train Loss" \
	     "Total Eval Loss(raw)" "Board Eval Loss (raw)" "Actions Eval Loss (raw)" \
	    "Total Train Loss(raw)" "Board Train Loss(raw)" "Actions Train Loss (raw)" ; do
	    printf '"%s" ' "${ii}" >> ${LOSS_DATA}
    done
    printf '\n' >> ${LOSS_DATA}

	cat "${LOGS_FILE}" \
	| perl -e '
	    $ma=0.90 ;
	    $t_m=0.0; $t2_m=0.0 ;
	    $b_m=0.0; $b2_m=0.0 ;
	    $a_m=0.0; $a2_m=0.0 ;
	    $c=0;
	 	while (<>) { 
	 		if ( m/.*Evaluation loss: total=(.*?) board=(.*?) actions=(.*)$/ ) {
	 			$t=$1; $b=$2; $a=$3 ;
	 			$r=$c/($c+1) ; if ($r>$ma) { $r=$ma } ; $r1=1.0-$r ; 
	 			$t_m=$t_m*$r+$t*$r1; 
	 			$b_m=$b_m*$r+$b*$r1; 
	 			$a_m=$a_m*$r+$a*$r1;
            } elsif ( m/.*Training loss: total=(.*?) board=(.*?) actions=(.*)$/ ) {
	 			$t2=$1; $b2=$2; $a2=$3 ;
	 			$r=$c/($c+1) ; if ($r>$ma) { $r=$ma } ; $r1=1.0-$r ;
	 			$t2_m=$t2_m*$r+$t2*$r1;
	 			$b2_m=$b2_m*$r+$b2*$r1;
	 			$a2_m=$a2_m*$r+$a2*$r1;
	 			$c++ ;
	 			print "$t_m $b_m $a_m $t2_m $b2_m $a2_m $t $b $a $t2 $b2 $a2\n";
	 		} 
	 	}' \
	>> "${LOSS_DATA}"
	gnuplot <<EOF
		set term png giant size 2048,1024 font '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

		set output '${LOSS_TOTAL_PNG}'
		plot for [i=1:6] "${LOSS_DATA}" using i title columnhead(i) with lines

		set output '${LOSS_BOARD_PNG}'
		plot "${LOSS_DATA}" using 2 title columnhead(2) with lines, \
		    "${LOSS_DATA}" using 5 title columnhead(5) with lines, \
		    "${LOSS_DATA}" using 8 title columnhead(8) with points, \
		    "${LOSS_DATA}" using 11 title columnhead(8) with points

		set output '${LOSS_ACTIONS_PNG}'
		plot "${LOSS_DATA}" using 3 title columnhead(3) with lines, \
		    "${LOSS_DATA}" using 6 title columnhead(6) with lines, \
		    "${LOSS_DATA}" using 9 title columnhead(9) with points, \
		    "${LOSS_DATA}" using 12 title columnhead(12) with points
EOF
	rm "${LOSS_DATA}"
}

function scores_scatter_plot() {
	SCORES_DATA="$(mktemp /tmp/scores_stats.XXXXXXXXX)"
	# printf '"Model" "Move" "Score"\n' > ${SCORES_DATA}

	cat "${LOGS_FILE}" \
    | egrep -i 'Move #' | cut -d']' -f2 | grep -v 'left)' \
    | perl -ne '/Move #(\d+) \((.*?)\).*score=(.*)$/ && print $1." ".$3." >".$2."\n";' \
    | perl -ne 's/>models.*$/1/g; s/>/0/g; print;' \
    | tail -n 5000 \
    >> ${SCORES_DATA}

	gnuplot <<EOF
		set term png giant size 2048,1024 font '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
		set output '${SCORES_PNG}'
		set style data points
		set xlabel "Move #"
		set ylabel "Score"
		set palette rgb 3,11
		plot("${SCORES_DATA}") with points palette
EOF
    rm "${SCORES_DATA}"
}

function diff_plot() {
	DIFF_DATA="$(mktemp /tmp/diff_stats.XXXXXXXXX)"
	printf '"Max-Min %%"\n' > ${DIFF_DATA}

	cat "${LOGS_FILE}" \
    | grep 'Difference' | cut -d']' -f2 \
	| perl -e '
	    $ma=0.99 ;
	    $d_m=0.0;
	    $c=0;
	 	while (<>) {
	 		if ( m/.*=(.*)%$/ ) {
	 			$d=$1 ;
	 			$r=$c/($c+1) ; if ($r>$ma) { $r=$ma } ; $r1=1.0-$r ;
	 			$d_m=$d_m*$r+$d*$r1;
	 			$c++ ;
	 			if ($c % 10 == 0) {
	 			    print "$d_m\n";
                }
	 		}
	 	}' \
    >> ${DIFF_DATA}

	gnuplot <<EOF
		set term png giant size 2048,1024 font '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
		set output '${DIFF_PNG}'
		plot for [i=1:1] "${DIFF_DATA}" using i title columnhead(i) with lines
EOF
    rm "${DIFF_DATA}"
}


wins_plot
loss_plot
if [[ "${SCORES_PNG}" != "" ]] ; then
    scores_scatter_plot
fi
if [[ "${DIFF_PNG}" != "" ]] ; then
    diff_plot
fi

