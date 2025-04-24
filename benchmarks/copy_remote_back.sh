#!/bin/bash

DIR=`dirname $0`

# Make your changes here !!!
REMOTE=user@192.168.15.167:/home/user/triton-benchmark/build-rv-0423/build-rope

# Make your changes here !!!
BUILD_DIR=${DIR}/build-rv-0423

scp -r ${REMOTE} ${BUILD_DIR}
