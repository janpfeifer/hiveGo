#!/bin/bash
BASELINE="linear"
MODEL="$1" ; shift

# Derived information
BASEDIR="models/${MODEL}"
BASE="${BASEDIR}/${MODEL}"
OUTPUT="${BASEDIR}/matches_${BASELINE}_vs_${MODEL}_checkpoint.png"

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

echo "${DATA}"
gnuplot <<EOF
	set term png giant size 2048,1024 font '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
	set output '${OUTPUT}'
	plot "${DATA}" using 1:(\$2+\$3+\$4) title columnhead(2) with filledcurves x1, \
		 "${DATA}" using 1:(\$3+\$4) title columnhead(3) with filledcurves x1, \
		 "${DATA}" using 1:(\$4) title columnhead(4) with filledcurves x1
EOF

rm -r "${DATA}"
cp -f "${OUTPUT}" "${HOME}/var/www"
