#!/bin/bash

# Constants
declare -a CHECKPOINT_FILES=("index" "data-00000-of-00001")
BASELINE="linear"
BASELINE_SPECS=""
NUM_MATCHES=100
DEPTH=2   # Same for both players.

# Randomness is applied only to the baseline player: since 
# the range of scores on the test player is chaning, it's hard
# to have a value that would be the same and comparable ... So
# baseline has a disadvanatage, but this is not an issue since 
# we care more about how test is changing over time.
RANDOMNESS=0.1

VLOG_LEVEL=0
VMODULE=''

# Program arguments
MODEL="$1" ; shift
CHECKPOINT="$1" ; shift

# Derived information
BASEDIR="models/${MODEL}"
BASE="${BASEDIR}/${MODEL}"
PB="${BASE}.pb"
if ! [[ -e "${PB}" ]] ; then
	echo "Model file '${PB}' doesn't exist!" 1>&2
	exit 1
fi

TEST_BASE="${BASEDIR}/test_${CHECKPOINT}"
TEST_PB="${TEST_BASE}.pb"
TEST_SPECS=",tf,model=${TEST_BASE}"
TEST_PARAMS="${BASEDIR}/tf_params.txt"

MATCH_BASE="match_${BASELINE}_vs_${CHECKPOINT}"
OUTPUT="${BASEDIR}/${MATCH_BASE}.txt"
MATCH_DIR="matches/${MODEL}"
mkdir -p "${MATCH_DIR}"
MATCH="${MATCH_DIR}/${MATCH_BASE}.bin"

# Copy checkpoint file to test 
cp "${PB}" "${TEST_PB}"
for ii in "${CHECKPOINT_FILES[@]}" ; do
	ln -f "${BASE}.checkpoint.${CHECKPOINT}.${ii}" "${TEST_BASE}.checkpoint.${ii}"
done

# Rebuild trainer fresh.
go install github.com/janpfeifer/hiveGo/trainer || (
	echo "Failed to build." 1>&2
	exit 1
)

# Train and measure times.
time trainer \
	--parallelism=50 --num_matches=${NUM_MATCHES} \
	--ai0="ab,max_depth=${DEPTH},${TEST_SPECS}" \
	--ai1="ab,max_depth=${DEPTH},randomness=${RANDOMNESS}${BASELINE_SPECS}" \
	--save_matches=${MATCH} \
	--v=${VLOG_LEVEL} --vmodule="${VMODULE}" --logtostderr \
	--tf_gpu_mem=0.1 --tf_params_file="${TEST_PARAMS}" \
	2>&1 \
	| tee ${OUTPUT} \
	| egrep --line-buffered '(finished at|Win|Draw)'

rm -f ${TEST_PB} 
for ii in "${CHECKPOINT_FILES[@]}" ; do
	rm -f "${TEST_BASE}.checkpoint.${ii}"
done
