import os 
import json 
from tqdm import tqdm 

BASE = "/largespace/tydata/code_optimization/cpp/saved_models/"
generated_model_id = "pie-gem5-by-user-cpp_deepseekcoder-7b_sft_moe-algorithm_0426"
generate_project = "generate_2_samples"

def is_answer_correct(input_case_path, stdout_bin):
    """
    check stdout_bin is correct 
    """
    output_case_path = input_case_path.replace("input", "output")
    with open(output_case_path, 'r') as g:
        truth = g.read()
    
    ground_truth_lines = truth.strip().splitlines()
    output_lines = stdout_bin.strip().splitlines()

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

def benchmark_postprocess(each_benchmark_path):
    # for each test binary benchmark dir
    benchmark_results_path = "testcases_3_benchmark_results.json"
    with open(os.path.join(each_benchmark_path, benchmark_results_path), 'r') as f:
        results = json.load(f)
    
    final_result = {   
        "binary_exec_right_and_gem5_right":{"testcases_id": [],"binary_exec_right_and_gem5_right_and_answer_correct":[], "binary_exec_right_and_gem5_right_and_answer_wrong":[]},
        "binary_exec_right_and_gem5_wrong":[],
        "binary_exec_wrong_and_gem5_right":[],
        "binary_exec_wrong_and_gem5_wrong":[],
        "testcases_number": 0
    }

    for testcase_result in results:
        returncode_bin = testcase_result["returncode_bin"]
        returncode_gem5 = testcase_result["returncode_gem5"]
        test_case_id = int(testcase_result["test_case_id"])
        
        if returncode_bin == 0 and returncode_gem5 == 0:
            final_result["binary_exec_right_and_gem5_right"]["testcases_id"].append(test_case_id)
            Is_answer_correct = is_answer_correct(testcase_result["input_case_path"], testcase_result["stdout_bin"])
            if Is_answer_correct:
                final_result["binary_exec_right_and_gem5_right"]["binary_exec_right_and_gem5_right_and_answer_correct"].append(test_case_id)
            else:
                final_result["binary_exec_right_and_gem5_right"]["binary_exec_right_and_gem5_right_and_answer_wrong"].append(test_case_id)
        elif returncode_bin == 0 and returncode_gem5 != 0:
            final_result["binary_exec_right_and_gem5_wrong"].append(test_case_id)
        elif returncode_bin != 0 and returncode_gem5 == 0:
            final_result["binary_exec_wrong_and_gem5_right"].append(test_case_id)
        else:
            final_result["binary_exec_wrong_and_gem5_wrong"].append(test_case_id)
        
        final_result["testcases_number"] += 1

    with open(os.path.join(each_benchmark_path, "analysis_result.json"), 'w') as writer:
        json.dump(final_result, writer, indent=4)


if __name__ == "__main__":
    benchmark_result_path = os.path.join(BASE, generated_model_id, generate_project, "benchmark_gem5_testcases_3")
    benchmarks = os.listdir(benchmark_result_path)

    for each_benchmark in tqdm(benchmarks):
        each_benchmark_path = os.path.join(benchmark_result_path, each_benchmark)
        benchmark_postprocess(each_benchmark_path)