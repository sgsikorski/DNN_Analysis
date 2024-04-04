#!/bin/bash

# Generate properties if asked to
if [ "$1" = "-gen" ]; then
    python3 generateProperties.py -o $2 -ic 50 -ec 10
fi

while IFS=',' read -ra array; do
  net+=("${array[0]}")
  spec+=("${array[1]}")
done < props/mnist_instances.csv 

#######################################
# AB-CROWN
#######################################
cd alpha-beta-CROWN/complete_verifier
# Change this to whatever your AB-CROWN conda environment is named
conda activate ab_crown

# Run ab-crown
for i in `seq 1 ${#spec[@]}`
do
    python abcrown.py --config configs/mnist.yaml
done

conda deactivate

#######################################
# NeuralSAT
#######################################
cd ../../neuralsat/neuralsat-pt201
# Change this to whatever your NeuralSat conda environment is named
conda activate neuralsat

# Run neuralSat
for i in `seq 1 ${#spec[@]}`
do
    python3 main.py --net ${net[i-1]} --spec ${spec[i-1]}
done