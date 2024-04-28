import argparse
import os 
import json
from tqdm import tqdm 
import resource
import shlex
import subprocess
import traceback
import re 
import glob
import multiprocessing

BASE = "/largespace/tydata/code_optimization/cpp/saved_models/"
generated_model_id = "pie-gem5-by-user-cpp_deepseekcoder-7b_sft_moe-algorithm_0426"
generate_project = "generate_2_samples"

def parse_args():
    parser = argparse.ArgumentParser(description="create and compile single python/c++ file")
    parser.add_argument('--cstd', type=str, default='std=c++17')
    parser.add_argument('--optimization_flag', type=str, default='-O3')
    parser.add_argument('--timeout_seconds_binary', type=int, default=10)
    parser.add_argument('--processing', type=str, default="single")
    args = parser.parse_args()
    
    return args

MAX_VIRTUAL_MEMORY = 10 * 1024 * 1024 * 50 # 500MB
def limit_virtual_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY*2, MAX_VIRTUAL_MEMORY*10))

def compile_cpp(args, cpp_file_path, target_dir):
    bin_file_path = re.sub('\.cpp', '.out', cpp_file_path)
    bin_file_path = re.sub('generate_cpp', 'generate_out', bin_file_path)

    cmd = f"g++ {cpp_file_path} -o {bin_file_path} --{args.cstd} {args.optimization_flag}"
    cmd_args = shlex.split(cmd)

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
        result['cpp_file_path'] = cpp_file_path
        result['bin_file_path'] = bin_file_path
        result['returncode'] = -100
        result['stdout'] = str(e)
        result['stderr'] = traceback.format_exc()
        result['exception'] = True
    
    return result

def compile(args):
    target_dir = os.path.join(BASE, generated_model_id, generate_project, "generate_out")
    if not os.path.exists(target_dir):
        print(f'{target_dir} dir does not exits, creating...')
        os.makedirs(target_dir, exist_ok=True)
    
    source_dir = os.path.join(BASE, generated_model_id, generate_project, "generate_cpp")
    cpp_files_path = glob.glob(os.path.join(source_dir, f"*.cpp"))

    compile_results = [] # List
    for single_cpp_file_path in tqdm(cpp_files_path):
        single_compile_result = compile_cpp(args, single_cpp_file_path, target_dir)
        compile_results.append(single_compile_result)

    # store compile results.
    results_path = os.path.join(BASE, generated_model_id, generate_project, f"compile_results.json")
    with open(results_path, 'w') as writer:
        json.dump(compile_results, writer, indent=4)


def parallel_compile(args):
    target_dir = os.path.join(BASE, generated_model_id, generate_project, "generate_out")
    if not os.path.exists(target_dir):
        print(f'{target_dir} dir does not exits, creating...')
        os.makedirs(target_dir, exist_ok=True)

    source_dir = os.path.join(BASE, generated_model_id, generate_project, "generate_cpp")
    cpp_files_path = glob.glob(os.path.join(source_dir, f"*.cpp"))

    pool = multiprocessing.Pool(processes=100)

    async_compile_results = [
        pool.apply_async(compile_cpp, args=(args, single_cpp_file_path, target_dir))
        for single_cpp_file_path in cpp_files_path
    ]

    compile_results = [result.get() for result in tqdm(async_compile_results)] # List

    # store compile results.
    results_path = os.path.join(BASE, generated_model_id, generate_project, f"compile_results.json")
    with open(results_path, 'w') as writer:
        json.dump(compile_results, writer, indent=4) 

if __name__ == "__main__":
    args = parse_args()

    if args.processing == "single":
        compile(args)
    elif args.processing == "multi":
        parallel_compile(args)
    else:
        raise ValueError(f"args processing must be in single or multi!")