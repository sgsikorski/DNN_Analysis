# this script runs NeuralSAT on the GTSRB dataset on their (30x30) network and saves the results to ns_gtsrb_output.txt

# NOTE: BEFORE running: run 'module load gurobi-11.0.0'
# NOTE: BEFORE running: run 'conda activate neuralsat'
# NOTE: run this from root directory

input_file="props/gtsrb/gtsrb_instances.csv"

output_file="logs/ns/ns_gtsrb_output.txt"
# > "$output_file"

while IFS=',' read -r col1 col2 col3 _ || [ -n "$col1" ]; do
    network="$col1"
    property="$col2"
    timeout="$col3"

    echo "Running $property..."

    python3 neuralsat/neuralsat-pt201/main.py --net props/gtsrb/$network --spec props/gtsrb/$property --timeout $timeout > temp_output.txt

    # Get the result (last line of the output) and write to output file
    result=$(tail -n 1 temp_output.txt)

    echo "$network,$property,$result" >> $output_file

    rm temp_output.txt

done < "$input_file"