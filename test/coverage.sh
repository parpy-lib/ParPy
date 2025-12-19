#!/bin/bash

# This script runs all relevant unit tests, integration tests, and example
# files, and produces a coverage report if all files work.
# This script runs both the unit tests in the native compiler and the
# integration tests in Python, and produces a coverage report based on this.
#
# In addition to the requirements for running ParPy, this file requires the
# 'llvm-cov' cargo package to be installed. Up to date installation
# instructions are found at:
# https://github.com/taiki-e/cargo-llvm-cov?tab=readme-ov-file#installation

run_test() {
  if [ "$#" -gt 1 ]; then
    output=$($1 2>&1)
  else
    eval "$1"
  fi
  return $?
}

cargo llvm-cov show-env --export-prefix --no-cfg-coverage > tmp.txt
source tmp.txt
rm tmp.txt
export CARGO_TARGET_DIR=$CARGO_LLVM_COV_TARGET_DIR
export CARGO_INCREMENTAL=1

cargo llvm-cov clean --workspace
maturin develop

fail_count=0

printf "Running unit tests\n"
run_test "cargo test"
fail_count=$((fail_count + $?))

printf "Running example files\n"
for f in "examples"/*."py"; do
  run_test "python $f" "ignore"
  fail=$?
  if [ $fail -eq 0 ]; then
    printf "$f \e[32mok\e[0m\n"
  else
    printf "$f \e[31mfailed\e[0m\n"
  fi
  fail_count=$((fail_count + $fail))
done

printf "\nRunning integration tests\n"
run_test "pytest"
fail_count=$((fail_count + $?))

cargo llvm-cov report

rm -f *profraw

if [ $fail_count -ne 0 ]; then
  exit 1
else
  exit 0
fi
