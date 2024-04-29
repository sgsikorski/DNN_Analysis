# this script runs NeuralSAT on the MNIST dataset on 256x6 network and saves the results to ns_mnist_256x6_output.txt

# NOTE: BEFORE running: run 'module load gurobi-11.0.0'
# NOTE: BEFORE running: run 'conda activate neuralsat'
# NOTE: run this from root directory


# remove existing vnnlib files
rm props/mnist_6/*.vnnlib

echo "Generating properties..."

# generate vnnlib files for model and dataset
python3 generateProperties.py -ic 10 -o props/mnist_6/mnist-net_256x6.onnx -s props/mnist_6/mnist_instances.csv -d MNIST_6 -ec 10 -se 0.0 -ee 0.04

run neuralsat on all instances

input_file="props/mnist_6/mnist_instances.csv"

output_file="logs/ns/ns_mnist_256x6_output.txt"
> "$output_file"

while IFS=',' read -r col1 col2 col3 _ || [ -n "$col1" ]; do
    network="$col1"
    property="$col2"
    timeout="$col3"

    echo "Running $property..."

    python3 neuralsat/neuralsat-pt201/main.py --net props/mnist_6/$network --spec props/mnist_6/$property --timeout $timeout > temp_output.txt

    # Get the result (last line of the output) and write to output file
    result=$(tail -n 1 temp_output.txt)

    echo "$network,$property,$result" >> $output_file

    rm temp_output.txt

done < "$input_file"

