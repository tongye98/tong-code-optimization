import argparse
import os 
import logging 
import json

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    filename='creating_xxx.log'  
)
# assert  1 == 2

TARGET_PROJECT = "/data1/tydata1/code_optimization/"
INPUT_PROJECT = "/home/tongye/code_generation/pie-perf/data/"

def parse_args():
    parser = argparse.ArgumentParser(description="create single python/c++ file")
    parser.add_argument('--output_dir', type=str, default=TARGET_PROJECT)
    parser.add_argument('--input_dir', type=str, default=INPUT_PROJECT)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--language', type=str, default='cpp')

    args = parser.parse_args()

    logging.info('Parsed arguments:')
    for arg_name, arg_value in args.__dict__.items():
        logging.info(f'{arg_name}: {arg_value}')
    
    return args

def creat_single_program(args):
    """
    make each pairs in train/val/test.jsonl to a seperate file.
    """
    target_split = os.path.join(args.output_dir, args.language, args.split)

    if not os.path.exists(target_split):
        logging.warning('{target_split} dir does not exits, creating...')
        os.makedirs(target_split, exist_ok=True)
    
    input_file = os.path.join(args.input_dir, f'{args.language}_splits', f'{args.split}.jsonl')

    data_points = []
    with open(input_file, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            data_points.append(json_data)

    logging.info(f"{input_file} has {len(data_points)} data points (pairs).")

    count = 0
    for item in data_points:
        user_id = item["user_id"]
        problem_id = item["problem_id"]
        submission_id_v0 = item["submission_id_v0"]
        submission_id_v1 = item["submission_id_v1"]

        input = item["input"]
        target = item["target"]

        # cpp name: problem_id | submission_id | user_id | cpp
        file_path_input = os.path.join(target_split, f"{problem_id}_{submission_id_v0}_{user_id}.cpp")
        if not os.path.exists(file_path_input):
            with open(file_path_input, 'w') as f:
                f.write(input)
                count += 1
        else:
            logging.info(f"cpp already exist: {file_path_input}")

        file_path_target = os.path.join(target_split, f"{problem_id}_{submission_id_v1}_{user_id}.cpp")
        if not os.path.exists(file_path_target):
            with open(file_path_target, 'w') as g:
                g.write(target)
                count += 1
        else:
            logging.info(f"cpp already exist: {file_path_target}")
    
    logging.info(f"Final {target_split} has {count} cpp files.")

    return None

def check_compile(args):
    """
    check each cpp file can be compiled by g++.
    """
    target_split = os.path.join(args.output_dir, args.language, f"{args.split}.out")
    # target_split: /data1/ytdata1/code_optimization/cpp/test_out
    if not os.path.exists(target_split):
        logging.warning('{target_split} dir does not exits, creating...')
        os.makedirs(target_split, exist_ok=True)
    return None 


import re 
from collections import defaultdict
def get_physical_cpu_list():
    cmd = " grep -E '^processor|^physical id|^core id' /proc/cpuinfo "
    output = os.popen(cmd).read()
    output = output.split("processor")
    output = [x for x in output if x]
    physical2logical = defaultdict(list)
    n_logical = 0
    for cpu_info in output:
        logical_id = re.search("(?<=\t: )\d+", cpu_info).group(0)
        physical_id = re.search("(?<=core id\t\t: )\d+", cpu_info).group(0)
        physical2logical[int(physical_id)].append(int(logical_id))
        n_logical += 1
    n_physical = len(physical2logical)
    from pprint import pformat
    logging.info(f"Physical CPU (n={n_physical}) to Logical CPU (n={n_logical}) mapping:")
    logging.info(pformat(sorted(dict(physical2logical).items(), key=lambda x: int(x[0]))))
    unique_logical_ids = []
    for physical_id, logical_ids in physical2logical.items():
        unique_logical_ids.append(logical_ids[0])
    logging.info(f"The set of logical ids available for use (n={len(unique_logical_ids)}):")
    logging.info(unique_logical_ids)
    return unique_logical_ids



if __name__ == "__main__":
    args = parse_args()

    # get_physical_cpu_list()

    # creat_single_program(args)

    check_compile(args)