import logging
import argparse 
import os 
import json 

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    filename='xxx.log'  
)

TARGET_PROJECT = "/data3/tydata3/code_optimization/"
INPUT_PROJECT = "/home/tongye/code_generation/pie-perf/data/"
MERGED_TEST_CASES = "/home/tongye/code_generation/pie-perf/data/test_cases/merged_test_cases/"
MAX_TESTCASES=3

def parse_args():
    parser = argparse.ArgumentParser(description="benchmark python/c++ out file")
    parser.add_argument('--output_dir', type=str, default=TARGET_PROJECT)
    parser.add_argument('--input_dir', type=str, default=INPUT_PROJECT)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--language', type=str, default='cpp')
    parser.add_argument('--cstd', type=str, default='std=c++17')
    parser.add_argument('--optimization_flag', type=str, default='-O3')
    parser.add_argument('--timeout_seconds_binary', type=int, default=10)
    parser.add_argument('--timeout_seconds_gem5', type=int, default=60)
    parser.add_argument('--gem5_opt', type=str, default='/home/tongye/code_generation/gem5/build/X86/gem5.opt')
    parser.add_argument('--gem5_script_path', type=str, default='/home/tongye/code_generation/gem5-skylake-config/gem5-configs/run-se.py')
    parser.add_argument('--cpu_type', type=str, default='Verbatim')

    args = parser.parse_args()

    logging.info('Parsed arguments:')
    for arg_name, arg_value in args.__dict__.items():
        logging.info(f'{arg_name}: {arg_value}')
    
    return args

def speedup_cal(slow:float, fast:float):
    return round(slow/fast, 4)

def relative_improve(slow:float, fast:float):
    return round((slow - fast) / slow, 4)


def gold_statistics(args):
    test_file = os.path.join(TARGET_PROJECT, args.language, "dataset", "by_user", f"{args.split}.json")
    with open(test_file, 'r') as f:
        datas = json.load(f)

    # print(len(datas))
    all_speedup = []
    count = 0
    count_no_improvement = 0
    count_improvment = 0
    for data_point in datas:
        slow_average_sim_seconds_precise = float(data_point["slow_average_sim_seconds_precise"])
        fast_average_sim_seconds_precise = float(data_point["fast_average_sim_seconds_precise"])
        speedup = speedup_cal(slow_average_sim_seconds_precise, fast_average_sim_seconds_precise)
        improvement = relative_improve(slow_average_sim_seconds_precise, fast_average_sim_seconds_precise)
        if improvement > 0.1:
            count_improvment += 1
            all_speedup.append(speedup)
        else:
            count_no_improvement  += 1
    
    average_speedup = round(sum(all_speedup) / len(all_speedup), 4)
    print(f"count improvement = {count_improvment}")
    print(f"count no improvemenet = {count_no_improvement}")
    print(f"average speedup = {average_speedup}")
    return None 


def gold2(args):
    file = os.path.join(TARGET_PROJECT, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "final_result_1sample_temp07_topp1.json")
    with open(file, 'r') as f:
        datas = json.load(f)

    all_speedup = []
    count = 0
    count_no_improvement = 0
    count_improvment = 0
    for data_point in datas:
        slow_average_sim_seconds_precise = float(data_point["slow_average_sim_seconds_precise"])
        fast_average_sim_seconds_precise = float(data_point["candidata_average_sim_seconds_precise"])
        if fast_average_sim_seconds_precise != -1:
            speedup = speedup_cal(slow_average_sim_seconds_precise, fast_average_sim_seconds_precise)
            improvement = relative_improve(slow_average_sim_seconds_precise, fast_average_sim_seconds_precise)
        else:
            speedup = 1.0
            improvement = 0.0

        if improvement > 0.1:
            count_improvment += 1
            all_speedup.append(speedup)
        else:
            all_speedup.append(speedup)
            count_no_improvement  += 1
    
    average_speedup = round(sum(all_speedup) / len(all_speedup), 4)
    print(f"count improvement = {count_improvment}")
    print(f"count no improvemenet = {count_no_improvement}")
    print(f"average speedup = {average_speedup}")
    return None 

if __name__ == "__main__":
    args = parse_args()

    # gold_statistics(args)

    gold2(args)