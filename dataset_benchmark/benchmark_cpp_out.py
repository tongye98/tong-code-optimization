import argparse
import os 
import logging 
import json
import re
import glob
import time 
import traceback
import subprocess
import shlex
import resource
import tqdm 

MAX_VIRTUAL_MEMORY = 10 * 1024 * 1024 * 50  # 500 MB
def limit_virtual_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY*2, MAX_VIRTUAL_MEMORY * 10))

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    filename='benchmark_xxx.log'  
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
    parser.add_argument('--timeout_seconds_gem5', type=int, default=120)
    parser.add_argument('--gem5_opt', type=str, default='/home/tongye/code_generation/gem5/build/X86/gem5.opt')
    parser.add_argument('--gem5_script_path', type=str, default='/home/tongye/code_generation/gem5-skylake-config/gem5-configs/run-se.py')
    parser.add_argument('--cpu_type', type=str, default='Verbatim')

    args = parser.parse_args()

    logging.info('Parsed arguments:')
    for arg_name, arg_value in args.__dict__.items():
        logging.info(f'{arg_name}: {arg_value}')
    
    return args

def execute_bin(args, bin_file_path, input_case_path):
    logging.info(f"Binary executing {bin_file_path}, with input {input_case_path}")
    with open(input_case_path, 'r') as fh:
        p = subprocess.run([bin_file_path],
                           preexec_fn=limit_virtual_memory,
                           bufsize=MAX_VIRTUAL_MEMORY,
                           capture_output=True,
                           timeout=args.timeout_seconds_binary,
                           stdin=fh,
                           text=True,
                           )
        returncode = p.returncode
        stdout = p.stdout
        stderr = p.stderr
    
    return returncode, stdout, stderr

def execute_gem5(args, bin_file_path, input_case_path, gem5_stats_out_path):
    cmd = f"{args.gem5_opt} --stats-file={gem5_stats_out_path} {args.gem5_script_path} {args.cpu_type} {bin_file_path}"
    logging.info(f'GEM5 executing {cmd}, with input {input_case_path}')
    cmd_args = shlex.split(cmd)
    with open(input_case_path, 'r') as fh:
        p = subprocess.run(cmd_args,
                           capture_output=True,
                           bufsize=MAX_VIRTUAL_MEMORY,
                           timeout=args.timeout_seconds_gem5,
                           stdin=fh,
                           text=True
                           )
        returncode = p.returncode
        stdout = p.stdout
        stderr = p.stderr
    
    return returncode, stdout, stderr


def benchmark_single_file(args, problem_id, bin_file_path, gem5_out_path):
    assert os.path.isdir(gem5_out_path),  f"{gem5_out_path} not exist!"

    test_cases_results = []
    test_cases_dir_in_problem_id = os.path.join(MERGED_TEST_CASES, problem_id)
    input_cases_paths = glob.glob(os.path.join(test_cases_dir_in_problem_id, 'input.*.txt'))
    # in_paths: list
    for input_case_path in input_cases_paths[:MAX_TESTCASES]: # for each input case
        # print(input_case_path)
        test_case_id = re.search('input\.([0-9]+)\.txt', input_case_path).group(1)
        gem5_stats_out_path = os.path.join(gem5_out_path, f'gem5_stats.{test_case_id}.txt')

        start_time = time.time()

        returncode_bin = returncode_gem5 =  -1
        stdout_bin, stderr_bin, stdout_gem5, stderr_gem5 = '', '', '', ''

        try:
            returncode_bin, stdout_bin, stderr_bin = execute_bin(args, bin_file_path, input_case_path)
            if returncode_bin != 0:
                raise Exception(f"Binary execution FAILED for {bin_file_path} with {input_case_path} with stderr {stderr_bin}")
            
            returncode_gem5, stdout_gem5, stderr_gem5 = execute_gem5(args, bin_file_path, input_case_path, gem5_stats_out_path)
            if returncode_gem5 != 0:
                raise Exception(f"gem5 execution FAILED for {bin_file_path} with {input_case_path} with stderr {stderr_gem5}")
            
            logging.info(f"BINARY execution and GEM5 execution SUCCEEDED for {bin_file_path} with {input_case_path}")

        except Exception as e:
            stderr_bin += str(e) + traceback.format_exc()
            stderr_gem5 += str(e) + traceback.format_exc()
            logging.warning(f"Execution FAILED with exception {str(e)} for {bin_file_path} with {input_case_path}")

        end_time = time.time()
        test_case_result = {
            'bin_file_path': bin_file_path,
            'input_case_path': input_case_path,
            'returncode': 0 if (returncode_bin == 0 and returncode_gem5 == 0) else -1,
            'test_case_id': test_case_id,
            'returncode_bin': returncode_bin,
            'stdout_bin': stdout_bin,
            'stderr_bin': stderr_bin,
            'returncode_gem5': returncode_gem5,
            'stdout_gem5': stdout_gem5,
            'stderr_gem5': stderr_gem5,
            'gem5_stats_out_path': gem5_stats_out_path,
            'time': end_time - start_time,
        }
        test_cases_results.append(test_case_result)

    return test_cases_results


def benchmark(args):
    compile_result_json = os.path.join(args.output_dir, args.language, f"compile_result_{args.split}.json")
    logging.info(f"compile result json = {compile_result_json}")

    with open(compile_result_json, 'r') as f:
        compile_results = json.load(f)
    
    for item in tqdm.tqdm(compile_results, desc="Benchmarking:"):  # for each bin out
        # item: dict
        cpp_file_path = item["cpp_file_path"]
        bin_file_path = item["bin_file_path"]

        logging.info(f"cpp file path = {cpp_file_path}")
        logging.info(f"bin file path = {bin_file_path}")

        pattern_problem_id = r'p\d+'
        pattern_submission_id = r's\d+'
        pattern_user_id = r'u\d+'

        match_problem_id = re.search(pattern_problem_id, bin_file_path)
        match_submission_id = re.search(pattern_submission_id, bin_file_path)
        match_user_id = re.search(pattern_user_id, bin_file_path)

        if match_problem_id and match_submission_id and match_user_id:
            problem_id = match_problem_id.group()
            submission_id = match_submission_id.group()
            user_id = match_user_id.group()
        else:
            raise ValueError("problem_id or submission_id or user_id does not exist!")

        logging.info(f"Problem id = {problem_id} | User id = {user_id} | Submission id = {submission_id}")

        # creat dir to store gem5 output
        # code_optimization → cpp → benchmark_gem5 → train/val/test_out 
        # → problem_id → user_id → submission_id → stat_testcaseid.txt
        gem5_out_path = os.path.join(TARGET_PROJECT, args.language, "benchmark_gem5", f"{args.split}_out", problem_id, user_id, submission_id)
        if not os.path.isdir(gem5_out_path):
            os.makedirs(gem5_out_path)
            logging.info(f"Directory '{gem5_out_path}' created successfully.")
        else:
            logging.info(f"Directory '{gem5_out_path}' already exists.")

        test_cases_results = benchmark_single_file(args, problem_id, bin_file_path, gem5_out_path)
        # (test_cases_results)
        logging.info("*"*60)
        with open(os.path.join(gem5_out_path, f"testcases_{MAX_TESTCASES}_benchmark_results.json"), 'w') as g:
            json.dump(test_cases_results, g, indent=4)

    return None

if __name__ == "__main__":
    args = parse_args()

    benchmark(args)