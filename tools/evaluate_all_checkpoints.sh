#!/bin/bash

BASELINE="linear"
MODEL="$1" ; shift

# Derived information
BASEDIR="models/${MODEL}"
BASE="${BASEDIR}/${MODEL}"

# Look for available checkpoints.
ls "${BASEDIR}/${MODEL}.checkpoint".[0-9]*.index \
| cut -d'.' -f 3 \
| while read checkpoint ; do
	match_base="match_${BASELINE}_vs_${checkpoint}"
	output="${BASEDIR}/${match_base}.txt"
	if ! [[ -e "${output}" ]] ; then
		echo "Evaluating checkpoint ${checkpoint}"
		tools/evaluate_model_checkpoint.sh "${MODEL}" "${checkpoint}"
	else
		echo "Skipping checkpoint ${checkpoint}, already generated."
	fi
	echo
done
