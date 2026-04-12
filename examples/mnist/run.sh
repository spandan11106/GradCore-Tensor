#!/bin/bash

# Configuration
LIB_DIR="../../build"
INC_DIR="../../include"
BIN_DIR="./bin"

mkdir -p $BIN_DIR

# Linker flags: 
# -L points to root build folder
# -lgradientcore links the built .so
# -Wl,-rpath tells the OS where to find the .so at runtime
FLAGS="-O3 -mavx2 -mfma -fopenmp -I$INC_DIR -L$LIB_DIR -lgradientcore -Wl,-rpath,$LIB_DIR"

echo "Compiling MNIST Trainer..."
g++ train.cpp $FLAGS -o $BIN_DIR/train_mnist

echo "Compiling MNIST Inference..."
g++ inference.cpp $FLAGS -o $BIN_DIR/inference_mnist

echo "Build complete. Ensure mnist_train.csv and mnist_test.csv are in data/."