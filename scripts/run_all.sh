#!/bin/bash
set -e

# This script provides a unified interface to build, run, and report benchmarks
# for both x86 and RISC-V platforms. It automatically detects the target
# platform from the global configuration and executes the appropriate pipeline.

# Get the directory of the current script.
DIR=$(dirname "$(readlink -f "$0")")
cd "${DIR}" || exit 1

# ==========================================
# 1. Load Global Configuration
# ==========================================
if [ ! -f "global_config.sh" ]; then
  echo "Error: global_config.sh not found in the scripts directory!"
  exit 1
fi
source "global_config.sh"

echo "=================================================="
echo "          Triton Benchmark Pipeline"
echo "=================================================="
echo "Platform:     ${PLATFORM}"
echo "Benchmarks:   ${BENCHMARKS_LIST}"
echo "Threads:      ${THREADS_LIST}"
echo "=================================================="

# ==========================================
# 2. Main Execution Logic
# ==========================================

# Function to execute a command and exit if it fails.
run_step() {
  echo -e "\n>>> EXECUTING STEP: $1"
  eval "$2"
  if [ $? -ne 0 ]; then
    echo "Error: Step '$1' failed. Aborting."
    exit 1
  fi
  echo ">>> STEP SUCCEEDED: $1"
}

# --- x86 Execution Pipeline ---
run_x86() {
  echo -e "\nStarting x86 pipeline..."
  run_step "Build Benchmarks"     "./compile.sh"
  run_step "Run Benchmarks"       "./execute.sh"
  run_step "Generate Report"      "./report.sh"
  echo -e "\nx86 pipeline completed successfully."
  echo "Reports are available in the 'build-<benchmark_name>' directories."
}

# --- RISC-V Execution Pipeline ---
run_riscv() {
  echo -e "\nStarting RISC-V pipeline..."
  run_step "Build RISC-V Binaries"           "./compile.sh --platform rv"
  run_step "Copy Binaries to Remote"       "./copy_to_remote.sh"

  echo -e "\n>>> EXECUTING REMOTE STEPS on ${REMOTE_URL}"
  ssh "${REMOTE_URL}" "
    echo '--- Changing to remote directory: ${REMOTE_BASE} ---'
    cd '${REMOTE_BASE}' || exit 1

    echo '--- Running benchmarks on remote ---'
    ./execute.sh || { echo 'Remote execute.sh failed'; exit 1; }

    echo '--- Generating report on remote ---'
    ./report.sh || { echo 'Remote report.sh failed'; exit 1; }
  "
  if [ $? -ne 0 ]; then
    echo "Error: Remote execution failed. Aborting."
    exit 1
  fi
  echo ">>> REMOTE STEPS SUCCEEDED"

  run_step "Copy Reports from Remote"      "./copy_remote_back.sh"
  echo -e "\nRISC-V pipeline completed successfully."
  echo "Reports have been copied back to the local 'build-<benchmark_name>' directories."
}

# --- Main Case Statement ---
case "${PLATFORM}" in
  "x86")
    run_x86
    ;;
  "rv")
    run_riscv
    ;;
  *)
    echo "Error: Invalid platform '${PLATFORM}' specified in global_config.sh."
    echo "Please set PLATFORM to either 'x86' or 'rv'."
    exit 1
    ;;
esac

echo -e "\n=================================================="
echo "           All tasks completed."
echo "=================================================="
