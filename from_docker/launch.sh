# from /home/working_dir/gem5/launch.sh
#!/bin/bash

export PYTHONPATH="/home/gem5-skylake-config/gem5-configs/system/"

if [ $# -ne 4 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

output_dir=$1 #/home/working_dir/outputs/ or /home/all_outputs or other 
split_id=$2 #0
num_splits=$3 #13000
timeout_seconds_gem5=$4 #>20 recommended


data_csv="/home/working_dir/improvement_pairs_additional_metadata_unpivoted_5_10_23.csv"
testcases_dir="/home/pie-perf/data/codenet/merged_test_cases/"
cstd="std=c++17"
gem5_dir="/home/gem5/build/X86/"
gem5_script_path="/home/gem5-skylake-config/gem5-configs/run-se.py"
cpu_type="Verbatim"
working_dir="/home/working_dir/"
path_to_atcoder="/home/ac-library/"
timeout_seconds_binary=5


echo python3 /home/working_dir/compile_and_benchmark.py --data_csv $data_csv \
--output_dir=$output_dir \
--split_id=$split_id \
--num_splits=$num_splits \
--testcases_dir=$testcases_dir \
--cstd=$cstd \
--gem5_dir=$gem5_dir \
--gem5_script_path=$gem5_script_path \
--cpu_type=$cpu_type \
--working_dir=$working_dir \
--path_to_atcoder=$path_to_atcoder \
--timeout_seconds_binary=$timeout_seconds_binary \
--timeout_seconds_gem5=$timeout_seconds_gem5


python3 /home/working_dir/compile_and_benchmark.py --data_csv $data_csv \
--output_dir=$output_dir \
--split_id=$split_id \
--num_splits=$num_splits \
--testcases_dir=$testcases_dir \
--cstd=$cstd \
--gem5_dir=$gem5_dir \
--gem5_script_path=$gem5_script_path \
--cpu_type=$cpu_type \
--working_dir=$working_dir \
--path_to_atcoder=$path_to_atcoder \
--timeout_seconds_binary=$timeout_seconds_binary \
--timeout_seconds_gem5=$timeout_seconds_gem5