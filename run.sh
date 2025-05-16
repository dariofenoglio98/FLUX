#!/bin/bash

k_folds=$(python -c "from public.config import k_folds; print(k_folds)")
model_name=$(python -c "from public.config import model_name; print(model_name)")
dataset_name=$(python -c "from public.config import dataset_name; print(dataset_name)")
strategy=$(python -c "from public.config import strategy; print(strategy)")
drifting_type=$(python -c "from public.config import drifting_type; print(drifting_type)")
n_clients=$(python -c "from public.config import n_clients; print(n_clients)")
n_rounds=$(python -c "from public.config import n_rounds; print(n_rounds)")



# P(X)
non_iid_type='feature_skew_strict' 
for scaling in $(seq 1 8); do

    echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
        MODEL: $model_name\033[0m\n\033[1;36m \
        Dataset: $dataset_name\033[0m\n\033[1;36m \
        Strategy: $strategy\033[0m\n\033[1;36m \
        Drifting type: $drifting_type\033[0m\n\033[1;36m \
        Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
        Number of clients: $n_clients\033[0m\n\033[1;36m \
        Scaling: $scaling\033[0m\n\033[1;36m \
        Number of rounds: $n_rounds\033[0m\n \
        \033[1;36mK-Folds: $k_folds\033[0m\n"

    # K-Fold evaluation, if k_folds > 1
    for fold in $(seq 0 $(($k_folds - 1))); do        
        echo -e "\n\033[1;36mStarting fold $((fold + 1))\033[0m\n"

        # Clean and create datasets
        rm -rf data/cur_datasets/* 
        if [ "$dataset_name" == "CheXpert" ]; then
            python public/chexpert_data_gen.py --fold "$fold" --scaling "$scaling" --n_clients "$n_clients"
        else
            python public/generate_datasets.py --fold "$fold" --scaling "$scaling" --non_iid_type "$non_iid_type"
        fi

        cd "$strategy"
        python server.py --fold "$fold" &
        sleep 3  # Sleep for 2s to give the server enough time to start

        for i in $(seq 0 $(($n_clients - 1))); do
            echo "Starting client ID $i"
            python client.py --id "$i" --fold "$fold" &
        done

        # This will allow you to use CTRL+C to stop all background processes
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        # Wait for all background processes to complete
        wait

        # Clean up
        echo "Fold completed correctly"
        trap - SIGTERM 
        pkill -f client.py -9
        pkill -f sever.py -9

        # Change back to the root directory
        cd ..
        sleep 2

    done

    # K-Fold evaluation, if k_folds > 1
    if [ "$k_folds" -gt 1 ]; then

        echo -e "\n\033[1;36mAveraging the results of all folds\033[0m\n"
        # Averaging the results of all folds
        python public/average_results.py --scaling "$scaling" --non_iid_type "$non_iid_type"
        # Plot confidence interval plots
        # python public/plots_across_folds.py --dataset "$dataset_name"
    fi

    echo -e "\n\033[1;36mExperiment completed successfully\033[0m\n"

done



# P(Y)
non_iid_type='label_skew_strict' 
for scaling in $(seq 1 8); do

    echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
        MODEL: $model_name\033[0m\n\033[1;36m \
        Dataset: $dataset_name\033[0m\n\033[1;36m \
        Strategy: $strategy\033[0m\n\033[1;36m \
        Drifting type: $drifting_type\033[0m\n\033[1;36m \
        Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
        Number of clients: $n_clients\033[0m\n\033[1;36m \
        Scaling: $scaling\033[0m\n\033[1;36m \
        Number of rounds: $n_rounds\033[0m\n \
        \033[1;36mK-Folds: $k_folds\033[0m\n"

    # K-Fold evaluation, if k_folds > 1
    for fold in $(seq 0 $(($k_folds - 1))); do        
        echo -e "\n\033[1;36mStarting fold $((fold + 1))\033[0m\n"

        # Clean and create datasets
        rm -rf data/cur_datasets/* 
        if [ "$dataset_name" == "CheXpert" ]; then
            python public/chexpert_data_gen.py --fold "$fold" --scaling "$scaling" --n_clients "$n_clients"
        else
            python public/generate_datasets.py --fold "$fold" --scaling "$scaling" --non_iid_type "$non_iid_type"
        fi

        cd "$strategy"
        python server.py --fold "$fold" &
        sleep 3  # Sleep for 2s to give the server enough time to start

        for i in $(seq 0 $(($n_clients - 1))); do
            echo "Starting client ID $i"
            python client.py --id "$i" --fold "$fold" &
        done

        # This will allow you to use CTRL+C to stop all background processes
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        # Wait for all background processes to complete
        wait

        # Clean up
        echo "Fold completed correctly"
        trap - SIGTERM 
        pkill -f client.py -9
        pkill -f sever.py -9

        # Change back to the root directory
        cd ..
        sleep 2

    done



 # P(Y|X)
non_iid_type='label_condition_skew'
for scaling in $(seq 1 8); do

    echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
        MODEL: $model_name\033[0m\n\033[1;36m \
        Dataset: $dataset_name\033[0m\n\033[1;36m \
        Strategy: $strategy\033[0m\n\033[1;36m \
        Drifting type: $drifting_type\033[0m\n\033[1;36m \
        Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
        Number of clients: $n_clients\033[0m\n\033[1;36m \
        Scaling: $scaling\033[0m\n\033[1;36m \
        Number of rounds: $n_rounds\033[0m\n \
        \033[1;36mK-Folds: $k_folds\033[0m\n"

    # K-Fold evaluation, if k_folds > 1
    for fold in $(seq 0 $(($k_folds - 1))); do        
        echo -e "\n\033[1;36mStarting fold $((fold + 1))\033[0m\n"

        # Clean and create datasets
        rm -rf data/cur_datasets/* 
        if [ "$dataset_name" == "CheXpert" ]; then
            python public/chexpert_data_gen.py --fold "$fold" --scaling "$scaling" --n_clients "$n_clients"
        else
            python public/generate_datasets.py --fold "$fold" --scaling "$scaling" --non_iid_type "$non_iid_type"
        fi

        cd "$strategy"
        python server.py --fold "$fold" &
        sleep 3  # Sleep for 2s to give the server enough time to start

        for i in $(seq 0 $(($n_clients - 1))); do
            echo "Starting client ID $i"
            python client.py --id "$i" --fold "$fold" &
        done

        # This will allow you to use CTRL+C to stop all background processes
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        # Wait for all background processes to complete
        wait

        # Clean up
        echo "Fold completed correctly"
        trap - SIGTERM 
        pkill -f client.py -9
        pkill -f sever.py -9

        # Change back to the root directory
        cd ..
        sleep 2

    done



# P(X|Y)
non_iid_type='feature_condition_skew'
for scaling in $(seq 1 8); do

    echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
        MODEL: $model_name\033[0m\n\033[1;36m \
        Dataset: $dataset_name\033[0m\n\033[1;36m \
        Strategy: $strategy\033[0m\n\033[1;36m \
        Drifting type: $drifting_type\033[0m\n\033[1;36m \
        Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
        Number of clients: $n_clients\033[0m\n\033[1;36m \
        Scaling: $scaling\033[0m\n\033[1;36m \
        Number of rounds: $n_rounds\033[0m\n \
        \033[1;36mK-Folds: $k_folds\033[0m\n"

    # K-Fold evaluation, if k_folds > 1
    for fold in $(seq 0 $(($k_folds - 1))); do        
        echo -e "\n\033[1;36mStarting fold $((fold + 1))\033[0m\n"

        # Clean and create datasets
        rm -rf data/cur_datasets/* 
        if [ "$dataset_name" == "CheXpert" ]; then
            python public/chexpert_data_gen.py --fold "$fold" --scaling "$scaling" --n_clients "$n_clients"
        else
            python public/generate_datasets.py --fold "$fold" --scaling "$scaling" --non_iid_type "$non_iid_type"
        fi

        cd "$strategy"
        python server.py --fold "$fold" &
        sleep 3  # Sleep for 2s to give the server enough time to start

        for i in $(seq 0 $(($n_clients - 1))); do
            echo "Starting client ID $i"
            python client.py --id "$i" --fold "$fold" &
        done

        # This will allow you to use CTRL+C to stop all background processes
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        # Wait for all background processes to complete
        wait

        # Clean up
        echo "Fold completed correctly"
        trap - SIGTERM 
        pkill -f client.py -9
        pkill -f sever.py -9

        # Change back to the root directory
        cd ..
        sleep 2

    done


