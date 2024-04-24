# this script runs NeuralSAT on the MNIST dataset on eran network and saves the results to ns_mnist_eran_output.txt

# NOTE: run this before executing the script: conda activate neuralsat
# NOTE: run this from root directory

# remove existing vnnlib files
rm props/mnist/*.vnnlib

echo "Generating properties..."

# generate vnnlib files for model and dataset
python3 generateProperties.py --instanceCount 10 --onnxFile props/eran/ffnnSIGMOID__Point_6x200.onnx --specFile props/mnist/mnist_instances.csv --dataset MNIST --epsilonCount 12 -se 0.0 -ee 0.025

# run neuralsat on all instances

input_file="props/mnist/mnist_instances.csv"

output_file="logs/ns/ns_mnist_eran_output.txt"
> "$output_file"

while IFS=',' read -r col1 col2 col3 _ || [ -n "$col1" ]; do
    network="$col1"
    property="$col2"
    timeout="$col3"

    echo "Running $property..."

    python3 neuralsat/neuralsat-pt201/main.py --net props/eran/$network --spec props/mnist/$property --timeout $timeout > temp_output.txt

    # Get the result (last line of the output) and write to output file
    result=$(tail -n 1 temp_output.txt)

    echo "$network,$property,$result" >> $output_file

    rm temp_output.txt

done < "$input_file"