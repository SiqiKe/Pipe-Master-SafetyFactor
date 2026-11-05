#!/bin/bash

# Ensure we're in the pipe_network_master directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

source .venv/bin/activate


# For local machine: use sequential execution (remove --parallel) or reduce --num_processes
# Uncomment the line below if you want to use parallel processing with fewer cores

# Run the Python script for a 5-pipe network

python src/workflows/1_UDD_design.py \
    --base_swmm_folder data/processed/swmm_inputs \
    --udd_setup data/processed/setup.json \
    --network 5\
    --swmm_folder results/SOWs \
    --output_folder results/SOWs \
    --intensity_2yr_24hr 0.138 \
    --pipe_roughness_coef 0.012
echo "Finished running 1_UDD_design.py which generated the initial UDD design for the 5-pipe network."
# You can change the network number to run for different networks (e.g., 9, 16, 23, 37, 49, 62, 76, 84) by modifying the --network argument.
# This script can process multiple networks at the same time
# e.g., --network 5 9 16 23 37


python src/workflows/1_UDD_design_20prec.py \
    --base_swmm_folder data/processed/swmm_inputs \
    --udd_setup data/processed/setup.json \
    --network 5\
    --swmm_folder results/SOWs \
    --output_folder results/SOWs \
    --intensity_2yr_24hr 0.138 \
    --pipe_roughness_coef 0.012
echo "Finished running 1_UDD_design_20prec.py which generated the UDD design with 20% increase in precipitation intensity for the 5-pipe network."
# You can change the network number to run for different networks (e.g., 9, 16, 23, 37, 49, 62, 76, 84) by modifying the --network argument.
# This script can process multiple networks at the same time
# e.g., --network 5 9 16 23 37


python src/workflows/2_SOW_n_conv.py \
    --setup data/processed/setup.json \
    --prec_source_folder data/processed/precipitation/projection/results/ \
    --swmm_path results/SOWs/5/5_udd.inp \
    --samples_folder results/SOWs_conv_num \
    --network 5 \
    --with_eval \
    --num_SOWs 1 5 10 \
    --num_resample 3 3 3
    #--parallel \
    #--num_processes 6
echo "Finished running 2_SOW_n_conv.py which determined the number of SOWs needed for convergence in performance matrix for the 5-pipe network."
# We only test 1, 5, 10 SOWs for the 5-pipe network as an example. You may need to increase it to find the number of SOWs required for convergence.
# You can adjust the number of processes for local execution by modifying --num_processes.
# For a hpc, you can add --parallel and adjust --num_processes based on available cores.


python src/workflows/3_SOW_prep_udd_eval.py \
    --swmm_path results/SOWs \
    --networks 5\
    --setup data/processed/setup.json \
    --prec_source_folder data/processed/precipitation/projection/results/ \
    --samples_folder results/SOWs \
    --num_sample 10 \
    --with_eval \
    --rel_threshold 0.80 \
    --eval_20prec
    #--parallel \
    #--num_processes 6
echo "Finished running 3_SOW_prep_udd_eval.py which prepared SOWs and evaluated the UDD design for the 5-pipe network."
# We use the num_sample 10 (SOWs per scenario) for the 5-pipe network for the purpose of showing the workflow. You can change the number of samples based on previous convergence tests by modifying the --num_sample argument.
# You can adjust the number of processes for local execution by modifying --num_processes.
# For a hpc, you can add --parallel and adjust --num_processes based on available cores.

# python src/workflows/4_BruteForce.py \
#     --setup data/processed/setup.json \
#     --udd_results results/SOWs \
#     --network 5 \
#     --swmm_path results/SOWs \
#     --output BruteForce \
#     --num_total_sample 1000 \
#     --n_obj 4 \
#     --parallel \
#     --num_processes 6
# This function takes ~6 hours to complete for the 5-pipe network with 48 cores on a high-performance computing system. 
# We commented out this for quick start check.
# You can adjust the number of processes for local execution by modifying --num_processes.
# For a hpc, you can add --parallel and adjust --num_processes based on available cores.

python src/workflows/5_PipeByPipe.py \
    --setup data/processed/setup.json \
    --udd_results results/SOWs \
    --network 5 \
    --swmm_path results/SOWs \
    --output PipeByPipe \
    --n_obj 4 \
    --pop_size 100 \
    --tol 0.01 \
    --period 10 \
    --seed 1 \
    --gen 50
    #--parallel \
    #--num_processes 6
echo "Finished running 5_PipeByPipe.py which identified the approximately Pareto-optimal pipe-by-pipe design using NSGA-II for the 5-pipe network."
# You can adjust the number of processes for local execution by modifying --num_processes.
# For a hpc, you can add --parallel and adjust --num_processes based on available cores.


# # Example commends for convergence test for NSGA-II (PipeByPipe). We commented out this for quick start check.
# seeds=(1 2 3 4 5)
# for seed in "${seeds[@]}"; do
#     python src/workflows/5_PipeByPipe.py \
#         --setup data/processed/setup.json \
#         --udd_results results/SOWs \
#         --network 5 \
#         --swmm_path results/SOWs \
#         --output PipeByPipe_conv/seed_${seed} \
#         --n_obj 4 \
#         --num_processes 6 \
#         --pop_size 100 \
#         --tol 0.0075 \
#         --period 10 \
#         --seed $seed \
#         --gen 50
# done


python src/workflows/6_SystemSF.py \
    --setup data/processed/setup.json \
    --udd_results results/SOWs \
    --network 5 \
    --swmm_path results/SOWs \
    --output SystemSF \
    --n_obj 4 \
    --sf_range 0.5 3.0 \
    --num_points 200
    #--parallel \
    #--num_processes 6
echo "Finished running 6_SystemSF.py which identified the approximately Pareto-optimal system-wide safety factors for the 5-pipe network."
# You can adjust the number of processes for local execution by modifying --num_processes.
# For a hpc, you can add --parallel and adjust --num_processes based on available cores.


python src/workflows/7_FunctionSF_system.py \
    --setup data/processed/setup.json \
    --udd_results results/SOWs \
    --ssf_results sf_results_cleaned.csv \
    --network 5 \
    --swmm_path results/SOWs \
    --output FunctionSF_ssf \
    --n_obj 4 \
    --a_num 50 \
    --b_num 50 \
    --a_range -1.0 1.0 \
    --b_range -1.0 1.0 \
    --coarse_size 20 \
    --fine_size 100 \
    --later_size 100
    #--parallel \
    #--num_processes 6
echo "Finished running 7_FunctionSF_system.py which identified the approximately Pareto-optimal functional safety factors for the 5-pipe network."
# You can adjust the number of processes for local execution by modifying --num_processes.
# For a hpc, you can add --parallel and adjust --num_processes based on available cores.


# ## Example commends to run for all networks with 4 objectives, and varying fine_size based on network size. We commented out this for quick start check.
# NETWORKS=(5 9 16 23 37 49 62 76 84)

# for NETWORK in "${NETWORKS[@]}"; do
#     FINE_SIZE=$((20 * NETWORK))
#     python src/workflows/6_SystemSF.py \
#         --setup data/processed/setup.json \
#         --udd_results results/SOWs \
#         --network "$NETWORK" \
#         --swmm_path results/SOWs \
#         --output SystemSF \
#         --n_obj 4 \
#         --parallel \
#         --num_processes 48 \
#         --coarse_size 20 \
#         --fine_size "$FINE_SIZE" \
#         --later_size "$FINE_SIZE"
#     echo "Completed SystemSF for network $NETWORK, obj 4 (fine_size=$FINE_SIZE)"

#     python src/workflows/7_FunctionSF_system.py \
#         --setup data/processed/setup.json \
#         --udd_results results/SOWs \
#         --ssf_results sf_results_cleaned.csv \
#         --network "$NETWORK" \
#         --swmm_path results/SOWs \
#         --output FunctionSF_ssf \
#         --n_obj 4 \
#         --parallel \
#         --num_processes 48 \
#         --a_num 50 \
#         --b_num 50 \
#         --a_range 0.01 1.0 \
#         --b_range 0.01 1.0 \
#         --coarse_size 20 \
#         --fine_size "$FINE_SIZE" \
#         --later_size "$FINE_SIZE"
#     echo "Completed FunctionSF for network $NETWORK, obj 4 (fine_size=$FINE_SIZE)"
# done


python src/workflows/8_Sobol_analysis.py \
    --swmm_path results/SOWs \
    --setup data/processed/setup.json \
    --network 5\
    --params_bounds_file data/processed/para_bounds.json \
    --samples_folder results/SOWs_sobol \
    --num_sample 256 \
    --num_processes 2 \
    --seed 0
    # --per_scenario \
    # --num_sample_per_scenario 256
echo "Finished running 8_Sobol_analysis.py which performed the Sobol sensitivity analysis across different drivers based on the 5-pipe network."
# You can change the num_sample here. They normally should be a power of 2 (e.g., 1024, 2048). The larger the number, the more accurate the results will be. However, it will also take longer to run.
# You can run the analysis per scenario by using the --per_scenario flag and setup number of samples per scenario using --num_sample_per_scenario.
# You can adjust the number of processes for local execution by modifying --num_processes.
# For a hpc, you can add --parallel and adjust --num_processes based on available cores.

