#!/bin/bash
BASELINE="linear"
MODEL="$1" ; shift

# Derived information
BASEDIR="models/${MODEL}"
BASE="${BASEDIR}/${MODEL}"
OUTPUT="${BASEDIR}/matches_${BASELINE}_vs_${MODEL}_checkpoint.png"
LOSSES_DATA="${BASEDIR}/losses.dat"

DATA="$(mktemp "${BASEDIR}/plot_XXXXXX.dat")"
echo "CheckPoint P0 P1 Draw" > "${DATA}"
ls "${BASEDIR}/match_${BASELINE}_vs_"*".txt" \
| while read file_name ; do 
	checkpoint=$(echo ${file_name} | cut -d_ -f 5 | cut -d. -f1)
	if [[ "${checkpoint}" = "" ]] ; then
		continue
	fi
	content="$(egrep "(Win|Draw)" "${file_name}" | cut -f 2 | tr -d '%' | tr '\n' ' ')"
	if [[ "${content}" == "" ]] ; then
		continue
	fi
	printf "%s %s" "${checkpoint}" "${content}"
	echo
done >> "${DATA}"

# Losses data from distill files.
rm -f "${LOSSES_DATA}"
ls "${BASEDIR}/distil"*".txt" | while read file_name ; do
	cat "${file_name}" \
	| egrep '(checkpointing|Validation|Loss after epoch)' \
	| perl -e '
		$v=0.0; 
		$l=-1.0; 
		while (<>) { 
			if ( $l < 0.0 && /total=(.*?),/ ) { 
				$l = $1; 
			} elsif ( /Validation losses: (.*?),/ ) { 
				$v=$1; 
			} elsif ( /global_step=(\d+)/ ) { 
				print "$1 $v $l\n"; 
				$l=-1; 
			} 
		}' \
	>> "${LOSSES_DATA}"
done

if [[ -e "${LOSSES_DATA}" ]] ; then

gnuplot <<EOF
	set term png giant size 2048,1024 font '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
	set output "${OUTPUT}"
	set xlabel "Checkpoint"
	set ylabel "Experiment Wins / Draws / Baseline Wins"
	set y2label "Loss"
	set y2tics 0.05
	plot "${DATA}" using 1:(\$2+\$3+\$4) title columnhead(2) with filledcurves x1, \
		 "${DATA}" using 1:(\$4+\$3) title columnhead(4) with filledcurves x1, \
		 "${DATA}" using 1:(\$3) title columnhead(3) with filledcurves x1, \
		 "${LOSSES_DATA}" using 1:2 title "Validation Losses" with lines lw 3 axis x1y2, \
		 "${LOSSES_DATA}" using 1:3 title "Training Losses" with lines lw 3 axis x1y2
EOF

else
gnuplot <<EOF
	set term png giant size 2048,1024 font '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
	set output '${OUTPUT}'
	plot "${DATA}" using 1:(\$2+\$3+\$4) title columnhead(2) with filledcurves x1, \
		 "${DATA}" using 1:(\$4+\$3) title columnhead(4) with filledcurves x1, \
		 "${DATA}" using 1:(\$3) title columnhead(3) with filledcurves x1
EOF
fi


rm -r "${DATA}"
cp -f "${OUTPUT}" "${HOME}/var/www"
