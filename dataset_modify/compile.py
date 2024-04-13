import logging
import argparse 
import os 
import json 
import re 
import time 
from tqdm import tqdm
import ast 

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    filename='xxx2.log'  
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

def creat_single_program(args):
    """
    make each pairs in train/val/test.jsonl to a seperate file.
    """
    target_dir = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/test1")

    if not os.path.exists(target_dir):
        logging.warning('{target_split} dir does not exits, creating...')
        os.makedirs(target_dir, exist_ok=True)
    
    input_file = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "result_1sample_temp07_topp1.json")

    with open(input_file, 'r') as file:
        datas = json.load(file)

    logging.info(f"{input_file} has {len(datas)} data points (pairs).")

    count = 0
    for item in datas:
        user_id = item["slow_user_id"]
        problem_id = item["problem_id"]
        slow_submission_id = item["slow_submission_id"]
        fast_submission_id = item["fast_submission_id"]

        # input = item["input"]
        # target = item["target"]
        input = item["candidates_maybe_faster_code"][0]

        # cpp name: problem_id | user_id | submission_id | cpp
        file_path_input = os.path.join(target_dir, f"{problem_id}_{user_id}_{slow_submission_id}_{fast_submission_id}_maybe_fast.cpp")
        if not os.path.exists(file_path_input):
            with open(file_path_input, 'w') as f:
                f.write(input)
                count += 1
        else:
            logging.info(f"cpp already exist: {file_path_input}")

    print(f"count = {count}")
    return None

import resource
MAX_VIRTUAL_MEMORY = 10 * 1024 * 1024 * 50 # 500MB
def limit_virtual_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY*2, MAX_VIRTUAL_MEMORY*10))

import shlex
import subprocess
import traceback

def compile_cpp(args, cpp_file_path, target_out):
    bin_file_path = re.sub('\.cpp', '.out', cpp_file_path)
    bin_file_path = re.sub('_cpp', '_out', bin_file_path)

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
    target_out = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "1sample_temp07_topp1_out")
    # target_split_out: /data1/ytdata1/code_optimization/cpp/test_out
    if not os.path.exists(target_out):
        logging.warning(f'{target_out} dir does not exits, creating...')
        os.makedirs(target_out, exist_ok=True)
    
    source_project = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "1sample_temp07_topp1_cpp")
    cpp_files_path = glob.glob(os.path.join(source_project, f"*.cpp"))
    # print(cpp_files)
    # print(len(cpp_files))

    results = []
    cpp_files_count = len(cpp_files_path)
    for idx, cpp_file_path in enumerate(cpp_files_path):
        # each cpp_file compiled to cpp.out in 'test_out' dir and a result.json
        logging.info(f"Compiling {idx+1}/{cpp_files_count} {cpp_file_path}")
        print(f"Compiling {cpp_file_path}")
        result = compile_cpp(args, cpp_file_path, target_out)
        results.append(result)

    assert len(results) == cpp_files_count

    results_path = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "compile_result.json")
    with open(results_path, 'w') as fresult:
        json.dump(results, fresult, indent=4)

    return None 

import multiprocessing
def compile_check_multiprocess(args):
    """
    check each cpp file can be compiled by g++.
    """
    target_out = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "1sample_temp07_topp1_out")
    # target_split_out: /data1/ytdata1/code_optimization/cpp/test_out
    if not os.path.exists(target_out):
        logging.warning(f'{target_out} dir does not exits, creating...')
        os.makedirs(target_out, exist_ok=True)
    
    source_project = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "1sample_temp07_topp1_cpp")
    cpp_files_path = glob.glob(os.path.join(source_project, f"*.cpp"))
    # print(cpp_files)
    # print(len(cpp_files))
    cpp_files_count = len(cpp_files_path)
    results = []

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # for idx, cpp_file_path in enumerate(cpp_files_path):
        #     # each cpp_file compiled to cpp.out in 'test_out' dir and a result.json
        #     logging.info(f"Compiling {idx+1}/{cpp_files_count} {cpp_file_path}")
        #     # print(f"Compiling {cpp_file_path}")

            # result = compile_cpp(args, cpp_file_path, target_out)
            # results.append(result)
        result_features = [pool.apply_async(compile_cpp, args=(args, cpp_file_path, target_out))
                          for cpp_file_path in cpp_files_path]
        
        for idx, feature in enumerate(result_features):
            result = feature.get()
            results.append(result)
            logging.info(f"Finished compile {idx}/{cpp_files_count}")

    assert len(results) == cpp_files_count

    results_path = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "compile_result.json")
    with open(results_path, 'w') as fresult:
        json.dump(results, fresult, indent=4)

    return None 

def after_compile(args):
    target_out = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "1sample_temp07_topp1_out")
    binary_files_path = glob.glob(os.path.join(target_out, f"*.out"))
    print(f"bianry count = {len(binary_files_path)}")


######################################
######  BEGIN BENCHMARK  #############
######################################

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
        error_info = ''

        try:
            returncode_bin, stdout_bin, stderr_bin = execute_bin(args, bin_file_path, input_case_path)
            if returncode_bin != 0:
                raise Exception(f"Binary execution FAILED for {bin_file_path} with {input_case_path} with stderr {stderr_bin}")
            
            returncode_gem5, stdout_gem5, stderr_gem5 = execute_gem5(args, bin_file_path, input_case_path, gem5_stats_out_path)
            if returncode_gem5 != 0:
                raise Exception(f"gem5 execution FAILED for {bin_file_path} with {input_case_path} with stderr {stderr_gem5}")
            
            logging.info(f"BINARY execution and GEM5 execution SUCCEEDED for {bin_file_path} with {input_case_path}")

        except Exception as e:
            error_info = str(e) + traceback.format_exc()
            logging.warning(f"Execution FAILED with exception {str(error_info)} for {bin_file_path} with {input_case_path}")

        end_time = time.time()
        test_case_result = {
            'bin_file_path': bin_file_path,
            'input_case_path': input_case_path,
            'returncode': 0 if (returncode_bin == 0 and returncode_gem5 == 0) else -100,
            'test_case_id': test_case_id,
            'returncode_bin': returncode_bin,
            'stdout_bin': stdout_bin,
            'stderr_bin': stderr_bin,
            'returncode_gem5': returncode_gem5,
            'stdout_gem5': stdout_gem5,
            'stderr_gem5': stderr_gem5,
            'gem5_stats_out_path': gem5_stats_out_path,
            'time': end_time - start_time,
            'error_info': error_info,
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
        if item["returncode"] != 0:
            continue

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

import multiprocessing
from functools import partial 
def process_single_item(args, item):
    # item: dict
    if item["returncode"] != 0:
        return 
        
    cpp_file_path = item["cpp_file_path"]
    bin_file_path = item["bin_file_path"]

    # logging.info(f"cpp file path = {cpp_file_path}")
    # logging.info(f"bin file path = {bin_file_path}")

    # pattern_problem_id = r'p\d+'
    # pattern_submission_id = r's\d+'
    # pattern_user_id = r'u\d+'
    pattern_problem_id = r"out/(p\d+)"
    pattern_submission_id = r's\d+'
    pattern_user_id = r'u\d+'


    # match_problem_id = re.search(pattern_problem_id, bin_file_path)
    # match_submission_id = re.findall(pattern_submission_id, bin_file_path)
    # match_user_id = re.search(pattern_user_id, bin_file_path)
    match_problem_id = re.findall(pattern_problem_id, bin_file_path)
    match_submission_id = re.findall(pattern_submission_id, bin_file_path)
    match_user_id = re.findall(pattern_user_id, bin_file_path)


    if match_problem_id and match_submission_id and match_user_id:
        problem_id = match_problem_id[0]
        slow_submission_id = match_submission_id[0]
        fast_submission_id = match_submission_id[1]
        user_id = match_user_id[0]
    else:
        raise ValueError("problem_id or submission_id or user_id does not exist!")


    logging.info(f"Problem id = {problem_id} | User id = {user_id} | slow Submission id = {slow_submission_id} | fast Submission id = {fast_submission_id} ")

    # creat dir to store gem5 output
    # code_optimization → cpp → benchmark_gem5 → train/val/test_out 
    # → problem_id → user_id → submission_id → stat_testcaseid.txt
    gem5_out_path = os.path.join(TARGET_PROJECT, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "benchmark_gem5", problem_id, user_id, f"{slow_submission_id}_{fast_submission_id}")
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

def benchmark_multiprocess(args):
    compile_result_json = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "compile_result.json")
    logging.info(f"compile result json = {compile_result_json}")

    with open(compile_result_json, 'r') as f:
        compile_results = json.load(f)

    # create multi process pool
    pool = multiprocessing.Pool(processes=96)
    process_single_item_partial = partial(process_single_item, args)

    # for item in tqdm.tqdm(compile_results, desc="Benchmarking:"):  # for each bin out
    #     pool.apply_async(process_single_item, args=(args,item))

    with tqdm(total=len(compile_results), desc="Benchmarking:") as pbar:
        for _ in pool.imap_unordered(process_single_item_partial, compile_results):
            pbar.update(1)
    
    # close
    pool.close()
    pool.join()

    return None

######################################
######  AFTER BENCHMARK  #############
######################################

def check_warning_after_benchmark(args):
    compile_result_json = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "compile_result.json")
    logging.info(f"compile result json = {compile_result_json}")

    with open(compile_result_json, 'r') as f:
        compile_results = json.load(f)
    
    testcases_error_count = 0
    item_error_count = 0
    returncode_bin_error_count = 0
    returncode_gem5_error_count = 0
    for item in tqdm(compile_results, desc="Check_warning_after_benchmark:"):  # for each bin out
        # item: dict
        if item["returncode"] != 0: # pass error *.out
            continue

        cpp_file_path = item["cpp_file_path"]
        bin_file_path = item["bin_file_path"]

        logging.info(f"cpp file path = {cpp_file_path}")
        logging.info(f"bin file path = {bin_file_path}")

        pattern_problem_id = r"out/(p\d+)"
        pattern_submission_id = r's\d+'
        pattern_user_id = r'u\d+'

        match_problem_id = re.findall(pattern_problem_id, bin_file_path)
        match_submission_id = re.findall(pattern_submission_id, bin_file_path)
        match_user_id = re.findall(pattern_user_id, bin_file_path)


        if match_problem_id and match_submission_id and match_user_id:
            problem_id = match_problem_id[0]
            slow_submission_id = match_submission_id[0]
            fast_submission_id = match_submission_id[1]
            user_id = match_user_id[0]
        else:
            raise ValueError("problem_id or submission_id or user_id does not exist!")


        logging.info(f"Problem id = {problem_id} | User id = {user_id} | slow Submission id = {slow_submission_id} | fast Submission id = {fast_submission_id} ")


        # creat dir to store gem5 output
        # code_optimization → cpp → benchmark_gem5 → train/val/test_out 
        # → problem_id → user_id → submission_id
        gem5_out_path = os.path.join(TARGET_PROJECT, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "benchmark_gem5", problem_id, user_id, f"{slow_submission_id}_{fast_submission_id}")
        with open(os.path.join(gem5_out_path, f"testcases_{MAX_TESTCASES}_benchmark_results.json"), 'r') as g:
            test_cases_results = json.load(g)

        is_item_error = False
        # print(test_cases_results)
        for each_result in test_cases_results:
            returncode = each_result['returncode']
            if returncode == -100:
                testcases_error_count += 1
                is_item_error = True
            
            returncode_bin = each_result['returncode_bin']
            returncode_gem5 = each_result['returncode_gem5']
            if returncode_bin != 0:
                returncode_bin_error_count += 1
            if returncode_gem5 != 0:
                returncode_gem5_error_count += 1
        
        if is_item_error:
            item_error_count += 1
    
    print(f"testcases error count: {testcases_error_count}")
    print(f"item error count : {item_error_count}")
    print(f"returncode bin error count : {returncode_bin_error_count}")
    print(f"returncode gem5 error count : {returncode_gem5_error_count}")


def get_accuracy(stdout, truth) -> bool:
    """
    Compare the output of the code with the ground truth.
    """
    ground_truth_lines = truth.strip().splitlines()
    output_lines = stdout.strip().splitlines()
    IsCorrect = True
    for gen_output, ground_truth_output in zip(output_lines, ground_truth_lines):
        is_corr = gen_output == ground_truth_output
        if not is_corr:
            try:
                gen_output = float(gen_output)
                ground_truth_output = float(ground_truth_output)
                is_corr = abs(gen_output - ground_truth_output) < 1e-3
            except:
                pass
        
        if not is_corr:
            IsCorrect = False
    
    return IsCorrect

def check_correctness_after_benchmark(args):
    # check the cpp out is correct ?
    compile_result_json = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "compile_result.json")
    logging.info(f"compile result json = {compile_result_json}")

    with open(compile_result_json, 'r') as f:
        compile_results = json.load(f)
    
    print(f"befor compile, there are  {len(compile_results)} out length.")
    compile_correct_count = 0

    bench_correct_exec_correct_count = 0
    bench_correct_exec_wrong_count = 0
    bench_wrong_exec_correct_count = 0
    bench_wrong_exec_wrong_count = 0

    for item in tqdm(compile_results, desc="Check_correctness_after_benchmark:"):  # for each bin out
        # item: dict
        if item["returncode"] != 0: # confirm can compile, pass error *.out
            continue
        compile_correct_count += 1
        
        cpp_file_path = item["cpp_file_path"]
        bin_file_path = item["bin_file_path"]

        pattern_problem_id = r"out/(p\d+)"
        pattern_submission_id = r's\d+'
        pattern_user_id = r'u\d+'

        match_problem_id = re.findall(pattern_problem_id, bin_file_path)
        match_submission_id = re.findall(pattern_submission_id, bin_file_path)
        match_user_id = re.findall(pattern_user_id, bin_file_path)


        if match_problem_id and match_submission_id and match_user_id:
            problem_id = match_problem_id[0]
            slow_submission_id = match_submission_id[0]
            fast_submission_id = match_submission_id[1]
            user_id = match_user_id[0]
        else:
            raise ValueError("problem_id or submission_id or user_id does not exist!")


        logging.info(f"Problem id = {problem_id} | User id = {user_id} | slow Submission id = {slow_submission_id} | fast Submission id = {fast_submission_id} ")


        test_cases_dir_in_problem_id = os.path.join(MERGED_TEST_CASES, problem_id)
        gem5_out_path = os.path.join(TARGET_PROJECT, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "benchmark_gem5", problem_id, user_id, f"{slow_submission_id}_{fast_submission_id}")
        with open(os.path.join(gem5_out_path, f"testcases_{MAX_TESTCASES}_benchmark_results.json"), 'r') as g:
            test_cases_results = json.load(g)
        

        def analysis(each_result):

            returncode = each_result["returncode"]
            bench_result = True if returncode == 0 else False

            test_case_id = each_result["test_case_id"]

            groundtruth = os.path.join(test_cases_dir_in_problem_id, f"output.{test_case_id}.txt") # ground truth
            stdout_bin = each_result["stdout_bin"]  # get the run result

            with open(groundtruth, 'r') as g:
                truth = g.read().strip()

            IsCorrect = get_accuracy(stdout_bin, truth)
            exec_result = True if IsCorrect else False

            return bench_result, exec_result

        out_bench_result = True
        out_exec_result = True
        for each_result in test_cases_results:
            bench_result, exec_result = analysis(each_result)
            if bench_result is False: out_bench_result = False
            if exec_result is False: out_exec_result = False
        
        if out_bench_result and out_exec_result:
            bench_correct_exec_correct_count += 1
        elif out_bench_result and (not out_exec_result):
            bench_correct_exec_wrong_count += 1
        elif (not out_bench_result) and out_exec_result:
            bench_wrong_exec_correct_count += 1
        else:
            bench_wrong_exec_wrong_count += 1

    print(f"after compile, there are  {compile_correct_count} out length.")
    print(f"bench correct and exec correct = {bench_correct_exec_correct_count}")
    print(f"bench correct but exec wrong = {bench_correct_exec_wrong_count}")
    print(f"bench wrong but exec correct = {bench_wrong_exec_correct_count}")
    print(f"bench wrong and exec wrong = {bench_wrong_exec_wrong_count}")

def collect_bench_correct_exec_correct_by_user(args):
    """
    collect bench correct & exec correct by author -> prepared/
    """
    compile_result_json = os.path.join(args.output_dir, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "compile_result.json")
    logging.info(f"compile result json = {compile_result_json}")

    with open(compile_result_json, 'r') as f:
        compile_results = json.load(f)
    
    print(f"befor compile, there are  {len(compile_results)} out length.")

    user_summary = os.path.join(TARGET_PROJECT, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "bench_gem5-2.json")
    gem5_out = []

    for item in tqdm(compile_results, desc="collect bench&exec correct by user:"):  # for each bin out
        # item: dict
        if item["returncode"] != 0: # confirm can compile, pass error *.out
            continue
        
        cpp_file_path = item["cpp_file_path"]
        bin_file_path = item["bin_file_path"]

        pattern_problem_id = r"out/(p\d+)"
        pattern_submission_id = r's\d+'
        pattern_user_id = r'u\d+'

        match_problem_id = re.findall(pattern_problem_id, bin_file_path)
        match_submission_id = re.findall(pattern_submission_id, bin_file_path)
        match_user_id = re.findall(pattern_user_id, bin_file_path)


        if match_problem_id and match_submission_id and match_user_id:
            problem_id = match_problem_id[0]
            slow_submission_id = match_submission_id[0]
            fast_submission_id = match_submission_id[1]
            user_id = match_user_id[0]
        else:
            raise ValueError("problem_id or submission_id or user_id does not exist!")


        logging.info(f"Problem id = {problem_id} | User id = {user_id} | slow Submission id = {slow_submission_id} | fast Submission id = {fast_submission_id} ")


        test_cases_dir_in_problem_id = os.path.join(MERGED_TEST_CASES, problem_id)
        gem5_out_path = os.path.join(TARGET_PROJECT, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "benchmark_gem5", problem_id, user_id, f"{slow_submission_id}_{fast_submission_id}")
        with open(os.path.join(gem5_out_path, f"testcases_{MAX_TESTCASES}_benchmark_results.json"), 'r') as g:
            test_cases_results = json.load(g)
        

        def analysis(each_result):

            returncode = each_result["returncode"]
            bench_result = True if returncode == 0 else False

            test_case_id = each_result["test_case_id"]

            groundtruth = os.path.join(test_cases_dir_in_problem_id, f"output.{test_case_id}.txt") # ground truth
            stdout_bin = each_result["stdout_bin"]  # get the run result

            with open(groundtruth, 'r') as g:
                truth = g.read().strip()

            IsCorrect = get_accuracy(stdout_bin, truth)
            exec_result = True if IsCorrect else False

            return bench_result, exec_result

        out_bench_result = True
        out_exec_result = True
        for each_result in test_cases_results:
            bench_result, exec_result = analysis(each_result)
            if bench_result is False: out_bench_result = False
            if exec_result is False: out_exec_result = False
        
        if out_bench_result and out_exec_result:
            # get the binary that  bench is right and exec is right
            # make dir  prepared / test / problem / user / xxx.cpp + problem_user_bench_gem5.json
            # 1. 把相应的cpp文件放到上面的相应的文件夹内;
            # 2. 把相应的 gem5_stat.*.txt 对应的时间信息抽取出来放到 problem_user_ben_gem5.json中;
            # [{submit_id: {problem:xxx, user:xxx, sim_senconds:xxx, sim_ticks:xxx, sim_seconds_precise:xxx}}]

            # cpp_file_target = os.path.join(TARGET_PROJECT, args.language, "prepared_by_user", args.split, problem_id, user_id)
            # if not os.path.isdir(cpp_file_target):
            #     os.makedirs(cpp_file_target)
            #     logging.info(f"Directory '{cpp_file_target}' created successfully.")
            # else:
            #     logging.info(f"Directory '{cpp_file_target}' already exists.")

            # shutil.copy(cpp_file_path, cpp_file_target)

            
            # if os.path.exists(user_summary):
            #     with open(user_summary, 'r') as f_exist:
            #         existing_data = json.load(f_exist)
            # else:
            #     existing_data = []

            gem5_stats_paths = glob.glob(os.path.join(gem5_out_path, 'gem5_stats.*.txt'))
            seconds_precise = []
            for gem5_stats_path in gem5_stats_paths:
                test_case_id = re.search('gem5_stats\.([0-9]+)\.txt', gem5_stats_path).group(1)
                stats = parse_stats_txt(gem5_stats_path)
                # stats_record = {
                #     "problem_id": problem_id,
                #     "user_id": user_id,
                #     "slow_submission_id": slow_submission_id,
                #     "fast_submission_id": fast_submission_id,
                #     "test_case_id": test_case_id,
                #     "sim_seconds": stats["sim_seconds"],
                #     "sim_ticks": stats["sim_ticks"],
                #     "sim_seconds_precise": stats["sim_seconds_precise"]
                    
                # }
                # gem5_out.append(stats_record)
                sim_seconds_precise = stats["sim_seconds_precise"]
                seconds_precise.append(sim_seconds_precise)
            
            stats_record = {
                f"{problem_id}_{user_id}_{slow_submission_id}_{fast_submission_id}": {"problem_id": problem_id,
                "user_id": user_id,
                "slow_submission_id": slow_submission_id,
                "fast_submission_id": fast_submission_id,
                "average_sim_seconds_precise": sum(seconds_precise)/len(seconds_precise)}
            }     
            gem5_out.append(stats_record)
        else:
            continue
    
    with open(user_summary, 'w') as f_write:
        json.dump(gem5_out, f_write, indent=4)

def parse_stats_txt(gem5_stats_path):
    with open(gem5_stats_path, 'r') as f:
        stats_lines = f.readlines()
    
    stats = {}
    for line in stats_lines:
        if line.strip() == '':
            continue 
        if "Begin" in line:
            continue
        if "End" in line:
            continue
        line = re.sub("#.*", "", line).strip() # remove comments
        parts = line.split()
        parts = [part.strip() for part in parts]
        if len(parts) > 2:
            value = parts[1:]
        elif len(parts) == 2:
            value = parts[1]
        else:
            logging.warning(f"could not parse line {line}")
            continue
        key = parts[0]
        if isinstance(value, str):
            try:
                value = value.replace("%", "").replace("nan", "None").replace("inf", "None").replace("-inf", "None")
                value = ast.literal_eval(value) if value != "None" else None
            except:
                logging.warning(f"could not parse value {value} for key {key}")
        elif isinstance(value, list):
            try:
                value = [v.replace("%", "").replace("nan", "None").replace("inf", "None").replace("-inf", "None") for v in value]
                value = [ast.literal_eval(v) if v!= "None" else None for v in value]
            except:
                logging.warning(f"could not parse value {value} for key {key}")
        
        stats[key] = value
    stats["sim_seconds_precise"] = calculate_sim_seconds(stats)
    return stats

def calculate_sim_seconds(stats):
    # more accurate than sim_seconds
    return float(stats["sim_ticks"]) / float(stats["sim_freq"])

def final(args):
    user_summary = os.path.join(TARGET_PROJECT, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "bench_gem5-2.json")
    with open(user_summary,'r') as f:
        data = json.load(f)
    print(len(data))

    user_dict = {}
    for item in data:
        key = list(item.keys())[0]
        value = item[key]
        user_dict[key] = value["average_sim_seconds_precise"]

    result = os.path.join(TARGET_PROJECT, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "result_1sample_temp07_topp1.json")
    with open(result, 'r') as g:
        data_result = json.load(g)
    print(len(data_result))

    final_result = []
    for idx in tqdm(data_result):
        problem_id = idx["problem_id"]
        user_id = idx["slow_user_id"]
        slow_submission_id = idx["slow_submission_id"]
        fast_submission_id = idx["fast_submission_id"]

        tt = f"{problem_id}_{user_id}_{slow_submission_id}_{fast_submission_id}"
        aa = user_dict.get(tt, -1)
        idx["candidata_average_sim_seconds_precise"] = aa

        final_result.append(idx)

    with open(os.path.join(TARGET_PROJECT, "saved_models/pie-gem5-by-user-cpp_codellama-13b-hf_sft_0406/generate/", "final_result_1sample_temp07_topp1.json"), 'w') as gg:
        json.dump(final_result, gg, indent=4)




if __name__ == "__main__":
    args = parse_args()

    # creat_single_program(args)

    # compile_check(args)

    # compile_check_multiprocess(args)

    # after_compile(args)

    # benchmark_multiprocess(args)

    # check_warning_after_benchmark(args)

    # check_correctness_after_benchmark(args)

    # collect_bench_correct_exec_correct_by_user(args)

    final(args)