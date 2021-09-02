#!/bin/bash

export OPENCV_ROOT=/usr/include/opencv4
export NNSDK_DIR=/usr/lib
export DDK_DIR=/usr/share/npu/sdk
export TOOLCHAIN=
export CROSS_COMPILE=


make -f makefile-cv4.linux

