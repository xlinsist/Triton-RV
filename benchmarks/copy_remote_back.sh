#!/bin/bash

# This script copies the benchmark reports from the remote machine back to the local machine.

# Get the directory of the current script.
DIR=$(dirname "$(readlink -f "$0")")

# Load global configuration.
if [ ! -f "${DIR}/global_config.sh" ]; then
  echo "Error: global_config.sh not found in ${DIR}"
  exit 1
fi
source "${DIR}/global_config.sh"

# Convert the space-separated benchmark list into a bash array.
IFS=' ' read -r -a BENCHMARKS <<< "$BENCHMARKS_LIST"

echo "Copying reports from remote platform..."

for BENCHMARK in "${BENCHMARKS[@]}"; do
    # Define remote and local paths.
    REMOTE_REPORT_PATH="${REMOTE_BASE}/build-${BENCHMARK}/report.xls"
    REMOTE_TARGET="${REMOTE_URL}:${REMOTE_REPORT_PATH}"
    LOCAL_BUILD_DIR="${DIR}/build-${BENCHMARK}/"

    echo "  - Copying report for ${BENCHMARK} from ${REMOTE_TARGET} to ${LOCAL_BUILD_DIR}"

    # Ensure the local directory exists.
    mkdir -p "${LOCAL_BUILD_DIR}"

    # Use scp to copy the report file.
    if ! scp "${REMOTE_TARGET}" "${LOCAL_BUILD_DIR}"; then
        echo "Warning: Failed to copy report for ${BENCHMARK}. It might not exist on the remote."
    fi
done

echo "All reports have been processed."
