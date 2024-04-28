# from /home/working_dir/gem5/compile_and_benchmarking.py
import argparse
import pandas as pd
import shutil
import os
import warnings
import traceback
import logging
import subprocess
import glob
import re
import traceback
import time
import shlex

logging.basicConfig(level=logging.INFO)


def parse_args(): 
    ## parse arguments: path to csv, output dir, split id, number of splits, testcases_dir
    parser = argparse.ArgumentParser(description='Compile and benchmark gem5')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--split_id', type=int, help='split id')
    parser.add_argument('--num_splits', type=int, help='number of splits')
    parser.add_argument('--data_csv', type=str, help='path to csv file', default='data_csv="/home/working_dir/improvement_pairs_additional_metadata_unpivoted_5_10_23.csv"')
    parser.add_argument('--testcases_dir', type=str, help='testcases directory', default="/home/pie-perf/data/codenet/merged_test_cases/")
    parser.add_argument('--cstd', type=str, help='cstd', default='std=c++20')
    parser.add_argument('--optimization_flag', type=str, help='optimization', default='-O3')
    parser.add_argument('--gem5_dir', type=str, help='path containing gem5 binary and build', default='/home/gem5/build/X86/')
    parser.add_argument('--gem5_script_path', type=str, help='path to gem5 script', default='/home/gem5-skylake-config/gem5-configs/run-se.py')
    parser.add_argument('--cpu_type', type=str, help='cpu type', default='Verbatim')
    parser.add_argument('--working_dir', type=str, help='working directory', default='/home/workdir/')
    parser.add_argument('--path_to_atcoder', type=str, help='path to atcoder', default='/home/ac-library/')
    parser.add_argument('--timeout_seconds_binary', type=int, help='timeout seconds for binary', default=10)
    parser.add_argument('--timeout_seconds_gem5', type=int, help='timeout seconds for gem5', default=120)
    
    args = parser.parse_args()
    return args


def comple_cpp(args, code_path): 
    # compile
    bin_dir = os.path.dirname(code_path)
    bin_path = os.path.join(bin_dir, 'a.out')
    cmd = f"g++ {code_path} -o {bin_path} --{args.cstd} {args.optimization_flag}"
    logging.info(f'executing {cmd}')
    cmd_args = shlex.split(cmd)
    try: 
        p = subprocess.run(cmd_args, capture_output=True, timeout=args.timeout_seconds_binary, text=True)
    except Exception as e:
        return 1, '', str(e) + traceback.format_exc(), ""
    
    return p.returncode, p.stdout, p.stderr, bin_path


def exec_bin(args, bin_path, in_path):
    
    logging.info(f'executing {bin_path}, with input {in_path}')
    with open(in_path, 'r') as fh:
        p = subprocess.run([bin_path], capture_output=True, timeout=args.timeout_seconds_binary, stdin=fh, text=True)
    return p.returncode, p.stdout, p.stderr


def exec_gem5(args, bin_path, in_path, stats_out_path):
    gem5_output_dir = os.path.dirname(bin_path)
    gem5_bin = os.path.join(args.gem5_dir, 'gem5.opt')
    cmd = f"{gem5_bin} --stats-file={stats_out_path} {args.gem5_script_path} {args.cpu_type} {bin_path}"
    logging.info(f'executing {cmd}, with input {in_path}')
    cmd_args = shlex.split(cmd)
    with open(in_path, 'r') as fh:
        p = subprocess.run(cmd_args, capture_output=True, timeout=args.timeout_seconds_gem5, stdin=fh, text=True)
    return p.returncode, p.stdout, p.stderr
    


def run_benchmarks(args, bin_path, problem_id):
    
    test_results = []
    results_dir = os.path.dirname(bin_path)
    
    in_paths = glob.glob(os.path.join(args.testcases_dir, problem_id, 'input.*.txt'))
    for in_path in in_paths:
        tc_id = re.search('input\.([0-9]+)\.txt', in_path).group(1)
        stats_out_path = os.path.join(results_dir, f'stats.{tc_id}.txt')
        start = time.time()
        
        rc_bin = -1
        stdout_bin = ''
        stderr_bin = ''
        rc_gem5 = -1
        stdout_gem5 = ''
        stderr_gem5 = ''
        
        try:
            rc_bin, stdout_bin, stderr_bin = exec_bin(args, bin_path, in_path)
            if rc_bin != 0:
                raise Exception(f'binary execution failed for {bin_path} with {in_path} with stderr {stderr_bin}')
            
            rc_gem5, stdout_gem5, stderr_gem5 = exec_gem5(args, bin_path, in_path, stats_out_path)
            if rc_gem5 != 0:
                raise Exception(f'gem5 execution failed for {bin_path} with {in_path} with stderr {stderr_gem5}')
            
            logging.info(f'binary and gem5 execution succeeded for {bin_path} with {in_path}')
        
        except Exception as e:
            stderr_bin += str(e) + traceback.format_exc()
            stderr_gem5 += str(e) + traceback.format_exc()
            logging.warn(f'Execution failed with exception {str(e)} for {bin_path} with {in_path}')
        
        end = time.time()
        test_results.append({
            'rc': 0 if rc_bin == 0 and rc_gem5 == 0 else -1,
            'tc_id': tc_id,
            'rc_binary': rc_bin,
            'stdout_binary': stdout_bin,
            'stderr_binary': stderr_bin,
            'rc_gem5': rc_gem5,
            'stdout_gem5': stdout_gem5,
            'stderr_gem5': stderr_gem5,
            'bin_path': bin_path,
            'in_path': in_path,
            'stats_path': stats_out_path,
            'time': end - start,
        })


    return test_results


def compile_and_benchmark(args, code_path, problem_id):
    import os
    import shutil
    import logging
    import json
    
    results = {"code_path": code_path, 
               "compile_returncode": None,
               "compile_stdout": None,
               "compile_stderr": None,
               "test_results": None}
    
    # compile
    rc, stdout, stderr, bin_path = comple_cpp(args, code_path)
    results['compile_returncode'] = rc
    results['compile_stdout'] = stdout
    results['compile_stderr'] = stderr
    
    if rc == 0: 
        logging.info(f'compilation succeeded for {code_path}, binary at {bin_path}')
    else:
        logging.warn(f'compilation failed for {code_path} with stderr {stderr}')
        return results # exit early if compilation failed
        
    # run benchmarks
    test_results = run_benchmarks(args, bin_path, problem_id)
    results['test_results'] = test_results
    
    # log results
    results_dir = os.path.dirname(bin_path)
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(test_results, f)
    
    return results


def summarize_results(args, results, problem_id):
    n_success = 0
    n_tests = len(glob.glob(os.path.join(args.testcases_dir, problem_id, 'input.*.txt')))
    if results['compile_returncode'] == 0 and results['test_results'] is not None:
        for test_result in results['test_results']: 
            if test_result['rc'] == 0:
                n_success += 1
    return {'n_success': n_success, 'n_tests': n_tests, 'compile_returncode': results['compile_returncode']}


def main(args): 

    if args.cpu_type != 'Verbatim':
        raise NotImplementedError('Only Verbatim CPU is supported for now, other 2 seem to throw errors on our binaries')
    # symlink atcoder
    try: 
        os.symlink(args.path_to_atcoder, os.path.join(args.working_dir, 'atcoder'))
        compilation_dir = os.path.join(args.working_dir, 'atcoder')
    except FileExistsError:
        logging.warn('atcoder symlink already exists')
        compilation_dir = os.path.join(args.working_dir, 'atcoder')
    except Exception as e:
        logging.warn('atcoder symlink failed, using atcoder dir for working dir')
        traceback.print_exc()
        compilation_dir = args.path_to_atcoder
    # create output dir
    if not os.path.exists(args.output_dir):
        logging.warn('output dir does not exist, creating')
        os.makedirs(args.output_dir)
    # read data
    df = pd.read_csv(args.data_csv)
    # split data
    chunk_size = len(df) // args.num_splits
    index_start = args.split_id * chunk_size
    index_end = min(index_start + chunk_size, len(df))
    df_chunk = df.iloc[index_start:index_end]
    # make directories for each submission_id 
    output_dir_split = os.path.join(args.output_dir, f'split_{args.split_id}')
    code_paths = []
    new_rows = []
    for i, row in df_chunk.iterrows():
        submission_id = row['submission_id']
        code = row['code']
        submission_dir = os.path.join(output_dir_split, str(submission_id))
        if not os.path.exists(submission_dir):
            logging.info('creating submission dir: {}'.format(submission_dir))
            os.makedirs(submission_dir)
        else:
            logging.info('submission dir already exists: {}'.format(submission_dir))
        code_path = os.path.join(submission_dir, 'code.cpp')
        with open(code_path, 'w') as f:
            f.write(code)
            logging.info('wrote code to {}'.format(os.path.join(submission_dir, 'code.cpp')))
        row['code_path'] = code_path
        new_rows.append(row)
    df_chunk = pd.DataFrame(new_rows)
    # compile and benchmark loop
    all_results = []
    for i, row in df_chunk.iterrows():
        logging.info('compiling and benchmarking {}'.format(code_path))
        results = compile_and_benchmark(args, row['code_path'], row['problem_id'])
        summary = summarize_results(args, results, row['problem_id'])
        all_results.append({"submission_id": row['submission_id'],
                            "code_path": row['code_path'],
                            "problem_id": row['problem_id'],
                            "n_success": summary['n_success'],
                            "n_tests": summary['n_tests'],
                            "compile_returncode": summary['compile_returncode']})
    # write results
    logging.info('writing results to {}'.format(os.path.join(output_dir_split, 'results.csv')))
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(output_dir_split, 'results.csv'), index=False)
    print(df_results)
    
    return None


if __name__ == '__main__':
    args = parse_args()
    main(args)
        
        
        
    
    
    
    
    
    