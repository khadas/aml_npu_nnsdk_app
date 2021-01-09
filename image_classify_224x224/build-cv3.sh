#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "usage: $0 <linux sdk dir>"
    exit 1
fi


export AQROOT=$1
export SDK_DIR=$AQROOT/build/sdk
export NNSDK_DIR=$AQROOT/build/sdk/nn_sdk
export OPENCV_ROOT=$SDK_DIR/opencv3
export CROSS_COMPILE=aarch64-linux-gnu-
export TOOLCHAIN=$AQROOT/../../toolchains/gcc-linaro-aarch64-linux-gnu/bin
export LIB_DIR=$TOOLCHAIN/../aarch64-linux-gnu/libc/lib


make -f makefile-cv3.linux

