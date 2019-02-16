#!/bin/bash
MODEL="$1" ; shift
BASELINE="${1:-linear}" ; shift

MOVING_AVERAGE_MAX=0.80

# Derived information
BASEDIR="models/${MODEL}"
BASE="${BASEDIR}/${MODEL}"
PLAY_AND_TRAIN="${BASEDIR}/play_and_train.txt"
OUTPUT="${BASEDIR}/matches/${BASELINE}_vs_${MODEL}_checkpoint.png"
LOSSES_DATA="${BASEDIR}/losses.dat"

DATA="${BASEDIR}/matches_${BASELINE}.dat"
echo "CheckPoint P0 P1 Draw" > "${DATA}"
ls "${BASEDIR}/matches/${BASELINE}_vs_"*".txt" \
| while read file_name ; do 
	checkpoint=$(basename ${file_name} .txt | cut -d_ -f 3)
	if [[ "${checkpoint}" = "" ]] ; then
		continue
	fi
	content="$(egrep "(Win|Draw)" "${file_name}" \
	    | cut -f 2 | tr -d '%' | tr '\n' '\t')"
	if [[ "${content}" == "" ]] ; then
		continue
	fi
	printf "%s\t%s\n" "${checkpoint}" "${content}"
done \
| perl -e '
    $p0=0.0; $p1=0.0; $d_m=0.0;
    $c=0.0;
    while (<>) {
        @f = split/\t/;
        $c = $c + 1.0;
        $r = 1.0 - 1.0 / $c;
        if ( $r gt '${MOVING_AVERAGE_MAX}' ) {
          $r = '${MOVING_AVERAGE_MAX}';
        }
        $p0 = $r * $p0 + (1.0 - $r) * $f[1];
        $p1 = $r * $p1 + (1.0 - $r) * $f[2];
        $d = $r * $d + (1.0 - $r) * $f[3];
        print "$f[0] $p0 $p1 $d\n";
    }' \
>> "${DATA}"


rm -f "${LOSSES_DATA}"

function distill_losses() {
    # Losses data from distill files.
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
}

function play_and_train_losses() {
	cat "${PLAY_AND_TRAIN}" \
	| perl -e '
	    $ma='${MOVING_AVERAGE_MAX}' ;
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
	 		} elsif ( m/checkpointing at global_step=(\d+)/) {
	 		    $ckpt=$1;
	 			$c++ ;
	 			print "$ckpt $t_m $t2_m\n";
	 		}
	 	}' >> "${LOSSES_DATA}"
}
play_and_train_losses

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


# rm -r "${DATA}"
cp -f "${OUTPUT}" "${HOME}/var/www"
