#!/bin/bash
MODEL="$1" ; shift
BASELINE="${1:-linear}" ; shift

# Derived information
BASEDIR="models/${MODEL}"
BASE="${BASEDIR}/${MODEL}"

# Look for available checkpoints.
ls "${BASEDIR}/checkpoints/"[0-9]*.index \
| while read file ; do
    checkpoint="$(basename "${file}" .index)"
	match_base=""
	output="${BASEDIR}/matches/${BASELINE}_vs_${checkpoint}.txt"
	if ! [[ -e "${output}" ]] ; then
		echo "Evaluating checkpoint ${checkpoint}"
		tools/evaluate_model_checkpoint.sh "${MODEL}" "${checkpoint}" "${BASELINE}"
	else
		echo "Skipping checkpoint ${checkpoint}, already generated."
	fi
	echo
done
