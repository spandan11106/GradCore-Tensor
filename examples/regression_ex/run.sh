#!/bin/bash

LIB_DIR="../../build"
INC_DIR="../../include"
BIN_DIR="./bin"

mkdir -p $BIN_DIR

FLAGS="-O3 -mavx2 -mfma -fopenmp -I$INC_DIR -L$LIB_DIR -lgradientcore -Wl,-rpath,$LIB_DIR"

echo "Compiling Regression Trainer..."
g++ train.cpp $FLAGS -o $BIN_DIR/train_regression

echo "Compiling Regression Inference..."
g++ inference.cpp $FLAGS -o $BIN_DIR/inference_regression

echo "Done! Run ./bin/train_regression to start."
