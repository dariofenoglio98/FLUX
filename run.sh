#!/bin/bash

# This code is usually called from cross_validation.sh, and it starts the server and clients 
# for the federated learning process. The server is started first, and then the clients are started.

# import client_number from config.py
n_clients=$(python -c "from public.config import client_number; print(client_number)")
# strategy=$(python -c "from public.config import strategy; print(strategy)")
 
# run create data
# TODO shall display:
# Model, Dataset, Strategy, Non-IIDType, n_clients
cd fedavg

echo -e "\n\033[1;36mStarting server with model \033[0m\n"

# python one_shot_cluster_server.py &
# if [ $strategy == "FedAvg" ]; then
#     python server_FedAvg.py &
# elif [ $strategy == "dynamic_cluster_global_server" ]; then
#     python dynamic_cluster_global_server.py &
# else
#     echo "Invalid strategy"
#     exit 1
# fi
python server.py &
# python dynamic_cluster_global_server.py &
sleep 20  # Sleep for 2s to give the server enough time to start

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
