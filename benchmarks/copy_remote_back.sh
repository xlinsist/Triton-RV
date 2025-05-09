#!/bin/bash

DIR=`dirname $0`

# Make your changes here !!!
REMOTE=user@192.168.15.167:/home/user/triton-benchmark/build-rv/build-softmax

# Make your changes here !!!
BUILD_DIR=${DIR}/build-rv

scp -r ${REMOTE} ${BUILD_DIR}
