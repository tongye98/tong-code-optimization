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
from tqdm import tqdm 
import shutil
import ast 


MAX_VIRTUAL_MEMORY = 10 * 1024 * 1024 * 50  # 500 MB
def limit_virtual_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY*2, MAX_VIRTUAL_MEMORY * 10))

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    filename='generate_dataset_by_user_train.log'  
)

TARGET_PROJECT = "/data3/tydata3/code_optimization/"
INPUT_PROJECT = "/home/tongye/code_generation/pie-perf/data/"
MERGED_TEST_CASES = "/home/tongye/code_generation/pie-perf/data/test_cases/merged_test_cases/"
MAX_TESTCASES=3

def parse_args():
    parser = argparse.ArgumentParser(description="benchmark python/c++ out file")
    parser.add_argument('--output_dir', type=str, default=TARGET_PROJECT)
    parser.add_argument('--input_dir', type=str, default=INPUT_PROJECT)
    parser.add_argument('--split', type=str, default='train')
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

def benchmark_multiprocess(args):
    compile_result_json = os.path.join(args.output_dir, args.language, f"compile_result_{args.split}.json")
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
    compile_result_json = os.path.join(args.output_dir, args.language, f"compile_result_{args.split}.json")
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
        # → problem_id → user_id → submission_id
        gem5_out_path = os.path.join(TARGET_PROJECT, args.language, "benchmark_gem5", f"{args.split}_out", problem_id, user_id, submission_id)
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
    compile_result_json = os.path.join(args.output_dir, args.language, f"compile_result_{args.split}.json")
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

        test_cases_dir_in_problem_id = os.path.join(MERGED_TEST_CASES, problem_id)
        gem5_out_path = os.path.join(TARGET_PROJECT, args.language, "benchmark_gem5", f"{args.split}_out", problem_id, user_id, submission_id)
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
    compile_result_json = os.path.join(args.output_dir, args.language, f"compile_result_{args.split}.json")
    logging.info(f"compile result json = {compile_result_json}")

    with open(compile_result_json, 'r') as f:
        compile_results = json.load(f)
    
    print(f"befor compile, there are  {len(compile_results)} out length.")

    for item in tqdm(compile_results, desc="collect bench&exec correct by user:"):  # for each bin out
        # item: dict
        if item["returncode"] != 0: # confirm can compile, pass error *.out
            continue
        
        cpp_file_path = item["cpp_file_path"]
        bin_file_path = item["bin_file_path"]

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

        test_cases_dir_in_problem_id = os.path.join(MERGED_TEST_CASES, problem_id)
        gem5_out_path = os.path.join(TARGET_PROJECT, args.language, "benchmark_gem5", f"{args.split}_out", problem_id, user_id, submission_id)
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
            cpp_file_target = os.path.join(TARGET_PROJECT, args.language, "prepared_by_user", args.split, problem_id, user_id)
            if not os.path.isdir(cpp_file_target):
                os.makedirs(cpp_file_target)
                logging.info(f"Directory '{cpp_file_target}' created successfully.")
            else:
                logging.info(f"Directory '{cpp_file_target}' already exists.")

            shutil.copy(cpp_file_path, cpp_file_target)

            user_summary = os.path.join(cpp_file_target, "problem_user_bench_gem5.json")
            if os.path.exists(user_summary):
                with open(user_summary, 'r') as f_exist:
                    existing_data = json.load(f_exist)
            else:
                existing_data = []

            with open(user_summary, 'w') as f_write:
                gem5_stats_paths = glob.glob(os.path.join(gem5_out_path, 'gem5_stats.*.txt'))
                for gem5_stats_path in gem5_stats_paths:
                    test_case_id = re.search('gem5_stats\.([0-9]+)\.txt', gem5_stats_path).group(1)
                    stats = parse_stats_txt(gem5_stats_path)
                    stats_record = {
                        submission_id: {"problem_id": problem_id,
                                        "user_id": user_id,
                                        "submission_id": submission_id,
                                        "test_case_id": test_case_id,
                                        "sim_seconds": stats["sim_seconds"],
                                        "sim_ticks": stats["sim_ticks"],
                                        "sim_seconds_precise": stats["sim_seconds_precise"]
                                        }
                    }
                    existing_data.append(stats_record)

                json.dump(existing_data, f_write, indent=4)
        else:
            continue

def collect_bench_correct_exec_correct_by_problem(args):
    """
    collect bench correct & exec correct by problem -> prepared/
    """
    compile_result_json = os.path.join(args.output_dir, args.language, f"compile_result_{args.split}.json")
    logging.info(f"compile result json = {compile_result_json}")

    with open(compile_result_json, 'r') as f:
        compile_results = json.load(f)

    for item in tqdm(compile_results, desc="collect bench&exec correct by problem:"):  # for each bin out
        # item: dict
        if item["returncode"] != 0: # confirm can compile, pass error *.out
            continue
        
        cpp_file_path = item["cpp_file_path"]
        bin_file_path = item["bin_file_path"]

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

        test_cases_dir_in_problem_id = os.path.join(MERGED_TEST_CASES, problem_id)
        gem5_out_path = os.path.join(TARGET_PROJECT, args.language, "benchmark_gem5", f"{args.split}_out", problem_id, user_id, submission_id)
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
            cpp_file_target = os.path.join(TARGET_PROJECT, args.language, "prepared_by_problem", args.split, problem_id)
            if not os.path.isdir(cpp_file_target):
                os.makedirs(cpp_file_target)
                logging.info(f"Directory '{cpp_file_target}' created successfully.")
            else:
                logging.info(f"Directory '{cpp_file_target}' already exists.")

            shutil.copy(cpp_file_path, cpp_file_target)

            user_summary = os.path.join(cpp_file_target, "problem_bench_gem5.json")
            if os.path.exists(user_summary):
                with open(user_summary, 'r') as f_exist:
                    existing_data = json.load(f_exist)
            else:
                existing_data = []

            with open(user_summary, 'w') as f_write:
                gem5_stats_paths = glob.glob(os.path.join(gem5_out_path, 'gem5_stats.*.txt'))
                for gem5_stats_path in gem5_stats_paths:
                    test_case_id = re.search('gem5_stats\.([0-9]+)\.txt', gem5_stats_path).group(1)
                    stats = parse_stats_txt(gem5_stats_path)
                    stats_record = {
                        submission_id: {"problem_id": problem_id,
                                        "user_id": user_id,
                                        "submission_id": submission_id,
                                        "test_case_id": test_case_id,
                                        "sim_seconds": stats["sim_seconds"],
                                        "sim_ticks": stats["sim_ticks"],
                                        "sim_seconds_precise": stats["sim_seconds_precise"]
                                        }
                    }
                    existing_data.append(stats_record)

                json.dump(existing_data, f_write, indent=4)
        else:
            continue


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


def sort_submission_by_sim_seconds(bench, problem):
    """
    return:
        [{submission_id:xxx, user_id:xxx, problem_id:xxx, average_sim_seconds_precise:xxx}]
        in order: from big to small <==> from slow to fast
    """
    with open(bench, 'r') as f:
        data = json.load(f)
    
    times = {} # {submission_id1:[time1, time2, time3], submission_id2: [time1, time2, time3]}
    users = {}

    for entry in data:
        submission_id = list(entry.keys())[0]
        if submission_id not in times:
            times[submission_id] = []
            users[submission_id] = []
        
        times[submission_id].append(list(entry.values())[0]["sim_seconds_precise"])
        users[submission_id].append(list(entry.values())[0]["user_id"])
    
    
    # list is in order 
    average_time = {key: sum(value)/ len(value) for key, value in times.items()}
    # print(average_time)


    for key, value in users.items():
        # value is a list
        first_user_id = value[0]
        if not all(element == first_user_id for element in value):
            raise ValueError(f"same submission {key} belongs to different user {value}.")

    sorted_submission = sorted(average_time.items(), key=lambda x:x[1], reverse=True)
    # print(sorted_submission)

    result = []
    for (submission_id, sim_seconds) in sorted_submission:
        user_id = users[submission_id][0]
        item = {
            "problem_id": problem,
            "user_id": user_id,
            "submission_id": submission_id,
            "average_sim_seconds_precise": sim_seconds,
        }
        result.append(item)

    return result


def rank_by_problem(args):
    """
    generate a rank jsonl file in  /data3/tydata3/code_optimization/cpp/prepared_by_problem/test/
    rank.jsonl
    {submission_id:xxx, user_id:xxx, problem_id:xxx, average_sim_seconds_precise:xxx}
    """
    target_dir = os.path.join(TARGET_PROJECT, args.language, "prepared_by_problem", args.split)
    problem_dir = os.listdir(target_dir)
    print(f"There are {len(problem_dir)} problems.")
    # print(problem_dir)
    for each_problem in tqdm(problem_dir, desc="Rank"):
        problem_dir_path = os.path.join(target_dir, each_problem)
        # /data3/tydata3/code_optimization/cpp/prepared_by_problem/test/p03275
        bench = os.path.join(problem_dir_path, "problem_bench_gem5.json")
        result = sort_submission_by_sim_seconds(bench, each_problem)
        # [{submission_id:xxx, user_id:xxx, problem_id:xxx, average_sim_seconds_precise:xxx}]
        #  in order: from big to small <==> from slow to fast

        rank_jsonl_path = os.path.join(problem_dir_path, "rank.jsonl")
        with open(rank_jsonl_path, 'w') as g:
            for entry in result:
                json.dump(entry, g)
                g.write("\n")


def rank_by_user(args):
    """
    generate a rank jsonl file in /data3/tydata3/code_optimization/cpp/prepared_by_user/test/pxxxx/uxxxx/
    """
    target_dir = os.path.join(TARGET_PROJECT, args.language, "prepared_by_user", args.split)
    problem_dir = os.listdir(target_dir)
    print(f"There are {len(problem_dir)} problems.")
    for each_problem in tqdm(problem_dir,desc='Rank'):
        problem_dir_path = os.path.join(target_dir, each_problem)
        # /data3/tydata3/code_optimization/cpp/prepared_by_user/test/p03275
        user_dir = os.listdir(problem_dir_path)
        for each_user in user_dir:
            user_dir_path = os.path.join(problem_dir_path, each_user)
            # /data3/tydata3/code_optimization/cpp/prepared_by_user/test/p03275/u018679195
            bench = os.path.join(user_dir_path, "problem_user_bench_gem5.json")
            result = sort_submission_by_sim_seconds(bench, each_problem)

            rank_jsonl_path = os.path.join(user_dir_path, "rank.jsonl")
            with open(rank_jsonl_path, 'w') as g:
                for entry in result:
                    json.dump(entry, g)
                    g.write("\n")


def gem5_check_by_hand(args):
    bin_file_path = os.path.join("/data3/tydata3/code_optimization/cpp/test_out/p00030_s870509314_u032763525.out")
    input_case_path = os.path.join("/home/tongye/code_generation/pie-perf/data/test_cases/merged_test_cases/p00030/input.6.txt")

    cmd = f"{args.gem5_opt} --stats-file='temp.txt' {args.gem5_script_path} {args.cpu_type} {bin_file_path}"
    print(f'GEM5 executing {cmd}, with input {input_case_path}')
    cmd_args = shlex.split(cmd)
    with open(input_case_path, 'r') as fh:
        p = subprocess.run(cmd_args,
                           # preexec_fn=limit_virtual_memory,
                           capture_output=True,
                           # bufsize=MAX_VIRTUAL_MEMORY,
                           timeout=args.timeout_seconds_gem5,
                           stdin=fh,
                           text=True
                           )
        returncode = p.returncode
        stdout = p.stdout
        stderr = p.stderr

    # with open(input_case_path, 'r') as fh:
    #     p = subprocess.run([bin_file_path],
    #                        preexec_fn=limit_virtual_memory,
    #                        bufsize=MAX_VIRTUAL_MEMORY,
    #                        capture_output=True,
    #                        timeout=args.timeout_seconds_binary,
    #                        stdin=fh,
    #                        text=True,
    #                        )
    #     returncode = p.returncode
    #     stdout = p.stdout
    #     stderr = p.stderr


    print(f"returncode = {returncode}")
    print(f"stdout = {stdout}")
    print(f"stderr = {stderr}")

def statistics(args):
    cpp_file_count = 0
    target_dir = os.path.join(TARGET_PROJECT, args.language, "prepared_by_user", args.split)
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".cpp"):
                cpp_file_count += 1
    
    print(f"There are {cpp_file_count} cpp file.")

def make_pair(rank_file):
    """
    rank_file: rank.jsonl
    [a, b, c, d] -> [(a,b), (b,c), (c,d)]
    return:
        a list: [pair1, pair2, xxx]
    """
    rank_results = []
    with open(rank_file, 'r') as f:
        for line in f:
            data_point = json.loads(line)
            rank_results.append(data_point)
    # print(rank_results)

    # only one or zero submission_id
    if len(rank_results) < 2: return None 

    length = len(rank_results)
    pairs = []
    for i in range(length - 1):
        slow_id, fast_id = i, i+1

        data_slow = rank_results[slow_id]
        slow_cpp_file_name = f"{data_slow['problem_id']}_{data_slow['submission_id']}_{data_slow['user_id']}.cpp"
        slow_cpp_file_path = os.path.join(TARGET_PROJECT, args.language, args.split, slow_cpp_file_name)
        with open(slow_cpp_file_path, 'r') as f_slow:
            slow_cpp_content = f_slow.read()

        data_fast = rank_results[fast_id]
        fast_cpp_file_name = f"{data_fast['problem_id']}_{data_fast['submission_id']}_{data_fast['user_id']}.cpp"
        fast_cpp_file_path = os.path.join(TARGET_PROJECT, args.language, args.split, fast_cpp_file_name)
        with open(fast_cpp_file_path, 'r') as f_fast:
            fast_cpp_content = f_fast.read()

        item = {
            "problem_id": data_slow["problem_id"],
            "slow_user_id": data_slow["user_id"],
            "slow_submission_id": data_slow["submission_id"],
            "slow_average_sim_seconds_precise": data_slow["average_sim_seconds_precise"],
            "slow_code": slow_cpp_content,
            "fast_user_id": data_fast["user_id"],
            "fast_submission_id": data_fast["submission_id"],
            "fast_average_sim_seconds_precise": data_fast["average_sim_seconds_precise"],
            "fast_code": fast_cpp_content
        }
        pairs.append(item)

    return pairs

def generate_dataset_by_problem(args):
    """
    generate dataset by 'problem'
    [   
        {
          problem_id: xxx,
          slow_user_id:xxx,
          slow_submission_id: xxx,
          slow_average_sim_seconds_precise: xxx,
          slow_code: xxx,
          fast_user_id:xxx,
          fast_submission_id: xxx,
          fast_average_sim_seconds_precise: xxx,
          fast_code: xxx
        }
    ]
    """
    target_dir = os.path.join(TARGET_PROJECT, args.language, "prepared_by_problem", args.split)
    problem_dir = os.listdir(target_dir)
    items = []
    for each_problem in tqdm(problem_dir, desc="Pair"):
        problem_dir = os.path.join(target_dir, each_problem)
        rank_file = os.path.join(problem_dir, "rank.jsonl")
        pairs = make_pair(rank_file) # [item1, item2, xxx]
        if pairs == None: continue

        for item in pairs:
            items.append(item)
    
    target_path = os.path.join(TARGET_PROJECT, args.language, "dataset", "by_problem", f"{args.split}.json")
    with open(target_path, 'w') as f:
        json.dump(items, f, indent=4)

def generate_dataset_by_user(args):
    """
    generate dataset by 'user'
    [   
        {
          problem_id: xxx,
          slow_user_id:xxx,
          slow_submission_id: xxx,
          slow_average_sim_seconds_precise: xxx,
          slow_code: xxx,
          fast_user_id:xxx,
          fast_submission_id: xxx,
          fast_average_sim_seconds_precise: xxx,
          fast_code: xxx
        }
    ]
    """
    target_dir = os.path.join(TARGET_PROJECT, args.language, "prepared_by_user", args.split)
    problem_dir = os.listdir(target_dir)
    items = []
    for each_problem in tqdm(problem_dir, desc="Pair"):
        problem_dir = os.path.join(target_dir, each_problem)
        user_dir = os.listdir(problem_dir)
        for each_user in user_dir:
            rank_file = os.path.join(problem_dir, each_user, "rank.jsonl")
            pairs = make_pair(rank_file) # [item1, item2, xxx]
            if pairs == None: continue

            for item in pairs:
                items.append(item)
    
    target_path = os.path.join(TARGET_PROJECT, args.language, "dataset", "by_user", f"{args.split}.json")
    with open(target_path, 'w') as f:
        json.dump(items, f, indent=4)

if __name__ == "__main__":
    args = parse_args()

    # benchmark(args)

    # benchmark_multiprocess(args)

    # check_warning_after_benchmark(args)

    # check_correctness_after_benchmark(args)

    # collect_bench_correct_exec_correct_by_user(args)

    # collect_bench_correct_exec_correct_by_problem(args)

    # rank_by_problem(args)

    # rank_by_user(args)

    # gem5_check_by_hand(args)

    # statistics(args)
    
    # generate_dataset_by_problem(args)

    generate_dataset_by_user(args)