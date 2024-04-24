# this script runs NeuralSAT on the CIFAR dataset on resnet2b network and saves the results to ns_cifar_resnet2b_output.txt

# NOTE: run this before executing the script: conda activate neuralsat
# NOTE: run this from root directory

# remove existing vnnlib files
rm props/cifar/*.vnnlib

echo "Generating properties..."

# generate vnnlib files for model and dataset
python3 generateProperties.py --instanceCount 5 --onnxFile props/cifar/resnet_2b.onnx --specFile props/cifar/cifar_instances.csv --dataset CIFAR --epsilonCount 10 --startEpsilon 0.0 --endEpsilon 0.02

# run neuralsat on all instances

input_file="props/cifar/cifar_instances.csv"

output_file="logs/ns/ns_cifar_resnet2b_output.txt"
> "$output_file"

while IFS=',' read -r col1 col2 col3 _ || [ -n "$col1" ]; do
    network="$col1"
    property="$col2"
    timeout="$col3"

    echo "Running $property..."

    python3 neuralsat/neuralsat-pt201/main.py --net props/cifar/$network --spec props/cifar/$property --timeout $timeout > temp_output.txt

    # Get the result (last line of the output) and write to output file
    result=$(tail -n 1 temp_output.txt)

    echo "$network,$property,$result" >> $output_file

    rm temp_output.txt

done < "$input_file"

