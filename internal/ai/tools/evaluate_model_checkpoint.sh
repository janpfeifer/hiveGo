#!/bin/bash

# Constants
declare -a CHECKPOINT_FILES=("index" "data-00000-of-00001")
BASELINE="linear"
BASELINE_SPECS=""
NUM_MATCHES=100
DEPTH=3   # Same for both players.

# Randomness is applied only to the baseline player: since 
# the range of scores on the test player is changing, it's hard
# to have a value that would be the same and comparable ... So
# baseline has a disadvantage, but this is not an issue since
# we care more about how test is changing over time.
RANDOMNESS=0.1

VLOG_LEVEL=0
VMODULE=''

# Program arguments
export MODEL="$1" ; shift
export CHECKPOINT="$1" ; shift
BASELINE="$1" ; shift

# Derived information
BASEDIR="models/${MODEL}"
BASE="${BASEDIR}/${MODEL}"
PB="${BASE}.pb"
if ! [[ -e "${PB}" ]] ; then
	echo "Model file '${PB}' doesn't exist!" 1>&2
	exit 1
fi

# Copy model/checkpoint to target destination (base).
function cp_model() {
    checkpoint=$1 ; shift
    base=$1 ; shift
    cp "${PB}" "${base}.pb"
    for file_name in "${CHECKPOINT_FILES[@]}" ; do
        ln -f "${BASEDIR}/checkpoints/${CHECKPOINT}.${file_name}" \
            "${base}.checkpoint.${file_name}"
    done
}

# Prepare test model.
TEST_BASE="${BASEDIR}/test_${CHECKPOINT}"
TEST_SPECS=",tf,model=${TEST_BASE}"
TEST_PARAMS="${BASEDIR}/tf_params.txt"
cp_model "${CHECKPOINT}" "${TEST_BASE}"

if [[ "${BASELINE}" == "" ]] ; then
    BASELINE="linear"
    BASELINE_SPECS=""
else
    if [[ "${BASELINE}" =~ ^[0-9]*$ ]] ; then
        BASELINE_BASE="${BASEDIR}/baseline_${BASELINE}"
        BASELINE_SPECS=",tf,model=${BASELINE_BASE}"
        cp_model "${BASELINE}" "${BASELINE_BASE}"
    else 
        BASELINE_SPECS=",tf,model=models/${BASELINE}/${BASELINE}"
    fi
fi


MATCH_BASE="${BASELINE}_vs_${CHECKPOINT}"
OUTPUT="${BASEDIR}/matches/${MATCH_BASE}.txt"
MATCH_DIR="matches/${MODEL}"
mkdir -p "${MATCH_DIR}"
MATCH="${MATCH_DIR}/match_${MATCH_BASE}.bin"

# Rebuild trainer fresh.
go install github.com/janpfeifer/hiveGo/trainer || (
	echo "Failed to build." 1>&2
	exit 1
)

# Train and measure times.
time trainer \
	--parallelism=50 --num_matches=${NUM_MATCHES} \
	--ai0="ab,max_depth=${DEPTH},randomness=${RANDOMNESS}${TEST_SPECS}" \
	--ai1="ab,max_depth=${DEPTH},randomness=${RANDOMNESS}${BASELINE_SPECS}" \
	--save_matches=${MATCH} \
	--v=${VLOG_LEVEL} --vmodule="${VMODULE}" --logtostderr \
	--tf_gpu_mem=0.1 --tf_params_file="${TEST_PARAMS}" \
	2>&1 \
	| tee ${OUTPUT} \
	| egrep --line-buffered '(finished at|Win|Draw)'


function rm_model() {
    base=$1 ; shift
    rm -f "${base}.pb"
    for ii in "${CHECKPOINT_FILES[@]}" ; do
        rm -f "${base}.checkpoint.${ii}"
    done
}

rm_model "${TEST_BASE}"
if [[ "${BASELINE}" =~ ^[0-9]*$ ]] ; then
    rm_model "${BASELINE_BASE}"
fi
