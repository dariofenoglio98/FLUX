#!/bin/bash

model_name=$(python -c "from public.config import model_name; print(model_name)")
dataset_name=$(python -c "from public.config import dataset_name; print(dataset_name)")
strategy=$(python -c "from public.config import strategy; print(strategy)")
drifting_type=$(python -c "from public.config import drifting_type; print(drifting_type)")
non_iid_type=$(python -c "from public.config import non_iid_type; print(non_iid_type)")
n_clients=$(python -c "from public.config import n_clients; print(n_clients)")

echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
    MODEL: $model_name\033[0m\n\033[1;36m \
    Dataset: $dataset_name\033[0m\n\033[1;36m \
    Strategy: $strategy\033[0m\n\033[1;36m \
    Drifting type: $drifting_type\033[0m\n\033[1;36m \
    Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
    Number of clients: $n_clients\033[0m\n"

# Create datasets
python public/generate_datasets.py

# Change to the directory of the strategy
cd "$strategy"


# delete datasets and exit
cd ../data/cur_datasets
rm -rf *
exit

python server.py &
# python dynamic_cluster_global_server.py &
sleep 2  # Sleep for 2s to give the server enough time to start

for i in $(seq 1 $n_clients); do
    echo "Starting client ID $i"
    # python client_dynamic.py --id "$i" &
    python client.py --id "$i" &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
# Clean up
echo "Shutting down - processes completed correctly"
trap - SIGTERM && kill -- -$$
