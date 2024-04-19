import argparse
import os 
import logging 
import json
from tqdm import tqdm 

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    filename='compile.log'  
)

TARGET_PROJECT = "/largespace/tydata/code_optimization/cpp/saved_models/pie-gem5-by-user-cpp_deepseekcoder-7b_ft_0418/generate/generate_out/"
INPUT_PROJECT = "/largespace/tydata/code_optimization/cpp/saved_models/pie-gem5-by-user-cpp_deepseekcoder-7b_ft_0418/generate/generate_cpp/"
MODEL_DIR = "/largespace/tydata/code_optimization/cpp/saved_models/pie-gem5-by-user-cpp_deepseekcoder-7b_ft_0418/generate/"

def parse_args():
    parser = argparse.ArgumentParser(description="create and compile single python/c++ file")
    parser.add_argument('--output_dir', type=str, default=TARGET_PROJECT)
    parser.add_argument('--input_dir', type=str, default=INPUT_PROJECT)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--language', type=str, default='cpp')
    parser.add_argument('--cstd', type=str, default='std=c++17')
    parser.add_argument('--optimization_flag', type=str, default='-O3')
    parser.add_argument('--timeout_seconds_binary', type=int, default=10)

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


import resource
MAX_VIRTUAL_MEMORY = 10 * 1024 * 1024 * 50 # 500MB
def limit_virtual_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY*2, MAX_VIRTUAL_MEMORY*10))


import shlex
import subprocess
import traceback

def compile_cpp(args, cpp_file_path, target_dir):
    bin_file_path = re.sub('\.cpp', '.out', cpp_file_path)
    bin_file_path = re.sub('generate_cpp', 'generate_out', bin_file_path)


    cmd = f"g++ {cpp_file_path} -o {bin_file_path} --{args.cstd} {args.optimization_flag}"
    logging.info(f'Executing compile cmd: {cmd}')
    cmd_args = shlex.split(cmd)

    # stdout_path = re.sub('\.out','_compile_stdout.txt', bin_file_path)
    # stderr_path = re.sub('\.out','_compile_stderr.txt', bin_file_path)
    # print(f"stdout path = {stdout_path}")
    # print(f"stderr_path = {stderr_path}")
    result = {}
    try:
        p = subprocess.run(cmd_args,
                            preexec_fn=limit_virtual_memory,
                            bufsize=MAX_VIRTUAL_MEMORY,
                            timeout=args.timeout_seconds_binary,
                            text=True,
                            capture_output=True
                            )
        returncode = p.returncode
        stdout = p.stdout
        stderr = p.stderr

        result['cpp_file_path'] = cpp_file_path
        result['bin_file_path'] = bin_file_path
        result['returncode'] = returncode
        result['stdout'] = stdout
        result['stderr'] = stderr
        result['exception'] = False

    except Exception as e:
        logging.error(f"compilation error at {cpp_file_path}")
        result['cpp_file_path'] = cpp_file_path
        result['bin_file_path'] = bin_file_path
        result['returncode'] = -100
        result['stdout'] = str(e)
        result['stderr'] = traceback.format_exc()
        result['exception'] = True
    
    return result


import glob
def compile_check(args):
    """
    check each cpp file can be compiled by g++.
    """
    target_dir = os.path.join(args.output_dir)
    if not os.path.exists(target_dir):
        logging.warning(f'{target_dir} dir does not exits, creating...')
        os.makedirs(target_dir, exist_ok=True)
    
    source_project = os.path.join(args.input_dir)
    cpp_files_path = glob.glob(os.path.join(source_project, f"*.cpp"))

    results = []
    cpp_files_count = len(cpp_files_path)
    for cpp_file_path in tqdm(cpp_files_path):
        logging.info(f"Compiling {cpp_file_path}")
        result = compile_cpp(args, cpp_file_path, target_dir)
        results.append(result)

    assert len(results) == cpp_files_count

    results_path = os.path.join(MODEL_DIR, f"compile_result.json")
    with open(results_path, 'w') as fresult:
        json.dump(results, fresult, indent=4)

    return None 

def data_collection_after_compile(args):
    original_split_cpp = os.path.join(args.output_dir, args.language, args.split)
    cpp_files = glob.glob(os.path.join(original_split_cpp, "*.cpp"))
    print(f"Original {args.language}-{args.split} set has {len(cpp_files)} files.")

    target_split_out = os.path.join(args.output_dir, args.language, f"{args.split}_out")
    # target_split_out: /data1/ytdata1/code_optimization/cpp/test_out
    cpp_out_files = glob.glob(os.path.join(target_split_out, f"*.out"))
    print(f"After compile: {args.language}-{args.split} set has {len(cpp_out_files)} files.")
    print(f"Compiled Error files: {len(cpp_files)-len(cpp_out_files)} files.")

    compile_result_json = os.path.join(args.output_dir, args.language, f"compile_result_{args.split}.json")
    print(f"compile result json = {compile_result_json}")

    with open(compile_result_json, 'r') as f:
        compile_results = json.load(f) # list

    error_compile_count = 0
    exception_compile_count = 0
    for item in compile_results:
        # item: dict
        returncode = item["returncode"]
        if returncode != 0:
            error_compile_count += 1
            #TODO: can find which cpp is not compile correct.
            if returncode == -100:
                exception_compile_count += 1
                # print(item)
                # assert False

    print(f"ERROR compile count: {error_compile_count}")
    print(f"EXCEPTION compile count: {exception_compile_count}")

    return None 

def compile_by_hand(args):
    target_split = os.path.join(args.output_dir, args.language, args.split)
    # target_split_out: /data1/ytdata1/code_optimization/cpp/test
    cpp_file_path = os.path.join(target_split, "p03313_s860695382_u139031151.cpp")
    bin_file_path = "a.out"
    cmd = f"g++ {cpp_file_path} -o {bin_file_path} --{args.cstd} {args.optimization_flag}"
    cmd_args = shlex.split(cmd)
    p = subprocess.run(cmd_args)
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

def fix_for_servers(args):
    compile_result_json = os.path.join(args.output_dir, args.language, f"compile_result_{args.split}.json")
    print(f"compile result json = {compile_result_json}")

    with open(compile_result_json, 'r') as f:
        compile_results = json.load(f)
    
    for item in compile_results:
        # item: dict
        cpp_file_path = item["cpp_file_path"]
        bin_file_path = item["bin_file_path"]

        # print(f"before fix cpp file path = {cpp_file_path}")
        # print(f"after fix  bin file path = {bin_file_path}")

        new_cpp_file_path = cpp_file_path.replace('/data1/tydata1', '/data3/tydata3')
        new_bin_file_path = bin_file_path.replace('/data1/tydata1', '/data3/tydata3')

        # print(f"before fix cpp file path = {new_cpp_file_path}")
        # print(f"after fix  bin file path = {new_bin_file_path}")

        item["cpp_file_path"] = new_cpp_file_path
        item["bin_file_path"] = new_bin_file_path

    results_path = os.path.join(args.output_dir, args.language, f"fix_compile_result_{args.split}.json")
    with open(results_path, 'w') as fresult:
        json.dump(compile_results, fresult, indent=4)


    return None


if __name__ == "__main__":
    args = parse_args()

    # get_physical_cpu_list()

    # creat_single_program(args)

    compile_check(args)

    # data_collection_after_compile(args)

    # compile_by_hand(args)

    # fix_for_servers(args)