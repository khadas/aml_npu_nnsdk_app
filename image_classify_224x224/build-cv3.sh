#!/bin/bash

set -e

if uname -a | grep "x86"; then
	if [ -z "$1" ]; then
		echo "usage: $0 <linux sdk dir>"
		exit 1
	fi

	export AQROOT=$1
	export SDK_DIR=$AQROOT/build/sdk
	export NNSDK_DIR=$AQROOT/build/sdk/nn_sdk//lib/lib64
	export OPENCV_ROOT=$SDK_DIR/opencv3
	export CROSS_COMPILE=aarch64-linux-gnu-
	export TOOLCHAIN=$AQROOT/../../toolchains/gcc-linaro-aarch64-linux-gnu/bin/
	export LIB_DIR=$TOOLCHAIN/../aarch64-linux-gnu/libc/lib
else
	export OPENCV_ROOT=/usr/include/opencv2
	export NNSDK_DIR=/usr/lib
	export DDK_DIR=/usr/share/npu/sdk
	export TOOLCHAIN=
	export CROSS_COMPILE=

fi

make -f makefile-cv3.linux

