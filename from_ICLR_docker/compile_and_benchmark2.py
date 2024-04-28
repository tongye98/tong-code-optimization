# from /home/working_dir/compile_and_benchmark.py
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
from google.cloud import storage
import tempfile
import resource 


MAX_VIRTUAL_MEMORY = 10 * 1024 * 1024 * 50  # 500 MB
def limit_virtual_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY*2, MAX_VIRTUAL_MEMORY * 10))

logging.basicConfig(level=logging.INFO)

MAX_TESTCASES=3

def parse_args(): 
    ## parse arguments: path to csv, output dir, split id, number of splits, testcases_dir
    parser = argparse.ArgumentParser(description='Compile and benchmark gem5')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--split_id', type=int, help='split id')
    parser.add_argument('--num_splits', type=int, help='number of splits')
    parser.add_argument('--split_id_offset', type=int, help='split id offset, what number to start counting splits from', default=0)
    parser.add_argument('--data_csv', type=str, help='path to csv file', default='data_csv="/home/working_dir/improvement_pairs_additional_metadata_unpivoted_5_10_23.csv"')
    parser.add_argument('--testcases_dir', type=str, help='testcases directory', default="/home/pie-perf/data/codenet/merged_test_cases/")
    parser.add_argument('--cstd', type=str, help='cstd', default='std=c++17')
    parser.add_argument('--optimization_flag', type=str, help='optimization', default='-O3')
    parser.add_argument('--gem5_dir', type=str, help='path containing gem5 binary and build', default='/home/gem5/build/X86/')
    parser.add_argument('--gem5_script_path', type=str, help='path to gem5 script', default='/home/gem5-skylake-config/gem5-configs/run-se.py')
    parser.add_argument('--cpu_type', type=str, help='cpu type', default='Verbatim')
    parser.add_argument('--working_dir', type=str, help='working directory', default='/home/workdir/')
    parser.add_argument('--path_to_atcoder', type=str, help='path to atcoder', default='/home/ac-library/')
    parser.add_argument('--successful_submissions_path', type=str, help='path to successful submissions txt file', default='/home/working_dir/successful_submissions.txt')
    parser.add_argument('--timeout_seconds_binary', type=int, help='timeout seconds for binary', default=10)
    parser.add_argument('--timeout_seconds_gem5', type=int, help='timeout seconds for gem5', default=120)
    
    args = parser.parse_args()
    return args

def upload_directory_to_gcs(local_directory, gcs_directory_path):
    # Create a client using default credentials
    # client = storage.Client()

    # # Get the bucket
    # bucket = client.get_bucket(destination_bucket)

    # for root, dirs, files in os.walk(local_directory):
    #     for file in files:
    #         local_path = os.path.join(root, file)
    #         relative_path = os.path.relpath(local_path, local_directory)
    #         gcs_path = os.path.join(gcs_directory, relative_path)
            
    #         blob = bucket.blob(gcs_path)
            
    #         blob.upload_from_filename(local_path)
            
    #         print(f'File {local_path} uploaded to {gcs_path}.')
    p = subprocess.run(['gsutil', '-m', 'cp', '-r', local_directory, gcs_directory_path], capture_output=True, text=True)
    logging.info(f'executed {" ".join(p.args)}, with return code {p.returncode}, stdout {p.stdout}, stderr {p.stderr}')
    return p.returncode, p.stdout, p.stderr

def upload_file_to_gcs(local_path, gcs_directory_path):
    # Create a client using default credentials
    # client = storage.Client()

    # # Get the bucket
    # bucket = client.get_bucket(destination_bucket)

    # basename = os.path.basename(local_path)
    # gcs_path = os.path.join(gcs_directory, basename)
    
    # blob = bucket.blob(gcs_path)
    
    # blob.upload_from_filename(local_path)
    
    # print(f'File {local_path} uploaded to {gcs_path}.')
    p = subprocess.run(['gsutil', 'cp', local_path, gcs_directory_path], capture_output=True, text=True)
    logging.info(f'executed {" ".join(p.args)}, with return code {p.returncode}, stdout {p.stdout}, stderr {p.stderr}')
    return p.returncode, p.stdout, p.stderr
            
            
def parse_bucket_name(gcs_path):
    if gcs_path.startswith('gs://'):
        path_without_prefix = gcs_path[len('gs://'):]
        bucket_name = path_without_prefix.split('/')[0]
        gcs_directory = '/'.join(path_without_prefix.split('/')[1:])
        return bucket_name, gcs_directory
    else:
        raise ValueError("Invalid GCS path. Must start with 'gs://'.")
            

def compile_cpp(args, code_path): 
    # compile
    bin_dir = os.path.dirname(code_path)
    bin_path = os.path.join(bin_dir, 'a.out')
    stdout_path = os.path.join(bin_dir, 'compile_stdout.txt')
    stderr_path = os.path.join(bin_dir, 'compile_stderr.txt')
    cmd = f"g++ {code_path} -o {bin_path} --{args.cstd} {args.optimization_flag}"
    logging.info(f'executing {cmd}')
    cmd_args = shlex.split(cmd)
    try: 
        with open(stdout_path, 'w') as stdout_fh, open(stderr_path, 'w') as stderr_fh:
            p = subprocess.run(cmd_args, 
                               preexec_fn=limit_virtual_memory,
                               bufsize=MAX_VIRTUAL_MEMORY,
                               timeout=args.timeout_seconds_binary,
                               stdout=stdout_fh,
                               stderr=stderr_fh, 
                               text=True)
            stdout_fh.flush()
            stderr_fh.flush()
        
        with open(stdout_path, 'r') as stdout_fh, open(stderr_path, 'r') as stderr_fh:
            stdout = stdout_fh.read()
            stderr = stderr_fh.read()
            
        logging.info(f'executed {cmd}, with return code {p.returncode}, stdout {stdout}, stderr {stderr}')
        
    except Exception as e:
        return 1, '', str(e) + traceback.format_exc(), ""
    
    return p.returncode, stdout, stderr, bin_path


# Start the subprocess and capture the output

import io 
def exec_bin(args, bin_path, in_path):
    # buffer = io.StringIO()
    logging.info(f'executing {bin_path}, with input {in_path}')
    stderr_path = os.path.join(os.path.dirname(bin_path), 'run_stderr.txt')
    stdout_path = os.path.join(os.path.dirname(bin_path), 'run_stdout.txt')
    with open(in_path, 'r') as stdin_fh, open(stderr_path, 'w') as stderr_fh, open(stdout_path, 'w') as stdout_fh:
        p = subprocess.run(
            [bin_path],
            preexec_fn=limit_virtual_memory,
            bufsize=MAX_VIRTUAL_MEMORY,
            stdin=stdin_fh,
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True, 
            timeout=args.timeout_seconds_binary
        )
        stderr_fh.flush()
        stdout_fh.flush()
    with open(stderr_path, 'r') as stderr_fh, open(stdout_path, 'r') as stdout_fh:
        stderr = stderr_fh.read()
        stdout = stdout_fh.read()
    return p.returncode, stdout, stderr


def exec_gem5(args, bin_path, in_path, stats_out_path):
    gem5_output_dir = os.path.dirname(bin_path)
    stderr_path = os.path.join(gem5_output_dir, 'gem5_stderr.txt')
    stdout_path = os.path.join(gem5_output_dir, 'gem5_stdout.txt')
    gem5_bin = os.path.join(args.gem5_dir, 'gem5.opt')
    cmd = f"{gem5_bin} --stats-file={stats_out_path} {args.gem5_script_path} {args.cpu_type} {bin_path}"
    logging.info(f'executing {cmd}, with input {in_path}')
    cmd_args = shlex.split(cmd)
    with open(in_path, 'r') as stdin_fh, open(stderr_path, 'w') as stderr_fh, open(stdout_path, 'w') as stdout_fh:
        p = subprocess.run(cmd_args, 
                        #    preexec_fn=limit_virtual_memory,
                           bufsize=MAX_VIRTUAL_MEMORY,
                           timeout=args.timeout_seconds_gem5, 
                           stdin=stdin_fh,
                           stdout=stdout_fh,
                           stderr=stderr_fh,
                           text=True)
        stderr_fh.flush()
        stdout_fh.flush()
    with open(stderr_path, 'r') as stderr_fh, open(stdout_path, 'r') as stdout_fh:
        stderr = stderr_fh.read()
        stdout = stdout_fh.read()
    return p.returncode, stdout, stderr
    


def run_benchmarks(args, bin_path, problem_id):
    
    test_results = []
    results_dir = os.path.dirname(bin_path)
    
    in_paths = glob.glob(os.path.join(args.testcases_dir, problem_id, 'input.*.txt'))
    
    for in_path in in_paths[:MAX_TESTCASES]: 
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
            logging.warn(f'Execution failed with exception {str(e)} for {bin_path} with {in_path}, stderr {stderr_bin}')
        
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
    
    results_dir = os.path.dirname(code_path)
    
    # compile
    rc, stdout, stderr, bin_path = compile_cpp(args, code_path)
    results['compile_returncode'] = rc
    results['compile_stdout'] = stdout
    results['compile_stderr'] = stderr
    
    # with open(os.path.join(results_dir, 'compile_stderr.txt'), 'w') as f:
    #     f.write(stderr)
    # with open(os.path.join(results_dir, 'compile_stdout.txt'), 'w') as f:
    #     f.write(stdout)
        
    if rc == 0: 
        logging.info(f'compilation succeeded for {code_path}, binary at {bin_path}')
    else:
        logging.warn(f'compilation failed for {code_path} with stderr {stderr}')
        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            json.dump(results, f)
        return results # exit early if compilation failed
        
    # run benchmarks
    test_results = run_benchmarks(args, bin_path, problem_id)
    results['test_results'] = test_results
    
    # log results
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(test_results, f)
        
    # run_stderr = '\n'.join([test_result['stderr_binary'] for test_result in test_results])
    # run_stdout = '\n'.join([test_result['stdout_binary'] for test_result in test_results])
    # gem5_stderr = '\n'.join([test_result['stderr_gem5'] for test_result in test_results])
    # gem5_stdout = '\n'.join([test_result['stdout_gem5'] for test_result in test_results])
    # with open(os.path.join(results_dir, 'run_stderr.txt'), 'w') as f:
    #     f.write(run_stderr)
    # with open(os.path.join(results_dir, 'run_stdout.txt'), 'w') as f:
    #     f.write(run_stdout)
    # with open(os.path.join(results_dir, 'gem5_stderr.txt'), 'w') as f:
    #     f.write(gem5_stderr)
    # with open(os.path.join(results_dir, 'gem5_stdout.txt'), 'w') as f:
    #     f.write(gem5_stdout)
    
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
    if not os.path.exists(args.output_dir) and not args.output_dir.startswith('gs://'):
        logging.warn('output dir does not exist, creating')
        os.makedirs(args.output_dir, exist_ok=True)
    # read data
    df = pd.read_csv(args.data_csv)
    # split data
    chunk_size = len(df) // args.num_splits
    index_start = (args.split_id + args.split_id_offset) * chunk_size
    index_end = min(index_start + chunk_size, len(df))
    df_chunk = df.iloc[index_start:index_end]
    df_chunk_pid = df_chunk[df_chunk["problem_id"] == "p02794"].reset_index() 
    ## concatenate first 2 rows with this one extra row
    df_chunk = pd.concat([df_chunk.iloc[:10], df_chunk_pid], axis=0)
    # read in the successful ids
    if args.successful_submissions_path is not None: 
        if not os.path.exists(args.successful_submissions_path):\
            logging.critical(f'successful submissions path {args.successful_submissions_path} does not exist')
        else: 
            with open(args.successful_submissions_path, 'r') as f:
                successful_ids = set(line.strip() for line in f.readlines())
                logging.info(f'found {len(successful_ids)} successful ids')
            # filter out successful ids
            df_chunk_new = df_chunk[~df_chunk['submission_id'].isin(successful_ids)]
            logging.info(f'filtered out {len(df_chunk) - len(df_chunk_new)} successful ids')
            df_chunk = df_chunk_new
    logging.info(f"all submission_ids matching are {df_chunk['submission_id']}")
    # make directories for each submission_id 
    # if args.output_dir.startswith('gs://'):
        # bucket_name, gcs_directory = parse_bucket_name(args.output_dir)
    output_dir_split = os.path.join(args.output_dir, f'split_{args.split_id + args.split_id_offset}')
    if not args.output_dir.startswith('gs://'): 
        os.makedirs(output_dir_split, exist_ok=True)
    code_paths = []
    new_rows = []
    for i, row in df_chunk.iterrows():
        submission_id = row['submission_id']
        logging.info(f"testcase submission id is {submission_id}")
        code = row['code']
        # submission_dir = os.path.join(output_dir_split, str(submission_id))            
        temp_submission_dir = tempfile.mkdtemp()
        # if not os.path.exists(submission_dir):
        #     logging.info('creating submission dir: {}'.format(submission_dir))
        #     os.makedirs(submission_dir, exist_ok=True)
        # else:
        #     logging.info('submission dir already exists: {}'.format(submission_dir))
        code_path = os.path.join(temp_submission_dir, 'code.cpp')
        with open(code_path, 'w') as f:
            f.write(code)
            logging.info('wrote code to {}'.format(os.path.join(temp_submission_dir, 'code.cpp')))
        row['code_path'] = code_path
        new_rows.append(row)
    df_chunk = pd.DataFrame(new_rows)
    # compile and benchmark loop
    all_results = []
    for i, row in df_chunk.iterrows():
        logging.info('compiling and benchmarking {}'.format(code_path))
        results = compile_and_benchmark(args, row['code_path'], row['problem_id'])
        ## copy to destination
        submission_dir = os.path.join(output_dir_split, row['submission_id'])
        if args.output_dir.startswith('gs://'):
            rc, stdout, stderr = upload_directory_to_gcs(os.path.dirname(row['code_path']), submission_dir)
            if rc != 0:
                logging.critical(f'uploading to gcs failed with return code {rc} and stderr {stderr} and stdout {stdout}')
        else: 
            # if not os.path.exists(submission_dir):
            #     logging.info('creating submission dir: {}'.format(submission_dir))
            #     os.makedirs(submission_dir, exist_ok=True)
            shutil.copytree(os.path.dirname(row['code_path']), submission_dir, dirs_exist_ok=True)
        shutil.rmtree(os.path.dirname(row['code_path'])) # delete temp dir
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
    if args.output_dir.startswith('gs://'):
        temp_csv_dir = tempfile.mkdtemp()
        temp_csv_path = os.path.join(temp_csv_dir, 'results.csv')
        df_results.to_csv(temp_csv_path, index=False)
        rc, stdout, stderr = upload_file_to_gcs(temp_csv_path, os.path.join(output_dir_split)) 
        if rc != 0:
            logging.critical(f'uploading to gcs failed with return code {rc} and stderr {stderr} and stdout {stdout}')
    else: 
        df_results.to_csv(os.path.join(output_dir_split, 'results.csv'), index=False)
    print(df_results)
    
    return None


if __name__ == '__main__':
    args = parse_args()
    main(args)
        
        
        
    
    
    
    
    
    