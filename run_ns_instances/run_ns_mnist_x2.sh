# this script runs NeuralSAT on the MNIST dataset on 256x2 network and saves the results to ns_mnist_256x2_output.txt

# NOTE: BEFORE running: run 'module load gurobi-11.0.0'
# NOTE: run this before executing the script: conda activate neuralsat
# NOTE: run this from root directory

# remove existing vnnlib files
rm props/mnist_2/*.vnnlib

echo "Generating properties..."

# generate vnnlib files for model and dataset
python3 generateProperties.py --instanceCount 40 --onnxFile props/mnist_2/mnist-net_256x2.onnx --specFile props/mnist_2/mnist_instances.csv --dataset MNIST_2 --epsilonCount 24 -se 0.00 -ee 0.055

# run neuralsat on all instances

input_file="props/mnist_2/mnist_instances.csv"

output_file="logs/ns/ns_mnist_256x2_output.txt"
> "$output_file"

while IFS=',' read -r col1 col2 col3 _ || [ -n "$col1" ]; do
    network="$col1"
    property="$col2"
    timeout="$col3"

    echo "Running $property..."

    python3 neuralsat/neuralsat-pt201/main.py --net props/mnist_2/$network --spec props/mnist_2/$property --timeout $timeout > temp_output.txt

    # Get the result (last line of the output) and write to output file
    result=$(tail -n 1 temp_output.txt)

    echo "$network,$property,$result" >> $output_file

    rm temp_output.txt

done < "$input_file"

