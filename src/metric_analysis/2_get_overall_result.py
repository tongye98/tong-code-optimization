import os 
import json 
import re 
import ast 
import glob 
from tqdm import tqdm 

BASE = "/largespace/tydata/code_optimization/cpp/saved_models/"
generated_model_id = "pie-gem5-by-user-cpp_deepseekcoder-7b_sft_moe-algorithm_0426"
generate_project = "generate_2_samples"

generate_out_path = os.path.join(BASE, generated_model_id, generate_project, "generate_out")
generate_outs = os.listdir(generate_out_path)
print(f"There are {len(generate_outs)} generate out.")


def get_identifier(each_generate_out):
    pattern_problem_id = r'p\d+'
    pattern_user_id = r'u\d+'
    pattern_submission_id = r's\d+'
    pattern_sample_id = r'maybe_faster_\d+'

    problem_id = re.findall(pattern_problem_id, each_generate_out)[0]
    user_id = re.findall(pattern_user_id, each_generate_out)[0]
    submission_id_two = re.findall(pattern_submission_id, each_generate_out)
    slow_submission_id = submission_id_two[0]
    fast_submission_id = submission_id_two[1]
    sample_id = re.findall(pattern_sample_id, each_generate_out)[0]

    return f"{problem_id}_{user_id}_{slow_submission_id}_{fast_submission_id}"

def result_statistics(corresponding_benchmark):
    analysis_result_path = os.path.join(corresponding_benchmark, "analysis_result.json")
    with open(analysis_result_path, "r") as reader:
        analysis_result = json.load(reader)

    flag_exec_right_gem5_right = False
    flag_all_right = False

    count_exec_right_gem5_right = len(analysis_result["binary_exec_right_and_gem5_right"]["testcases_id"])
    if count_exec_right_gem5_right == analysis_result["testcases_number"]:
        flag_exec_right_gem5_right = True

    count_all_right = len(analysis_result["binary_exec_right_and_gem5_right"]["binary_exec_right_and_gem5_right_and_answer_correct"])
    if count_all_right == analysis_result["testcases_number"]:
        flag_all_right = True
    
    return (flag_exec_right_gem5_right, flag_all_right)

def calculate_sim_seconds(stats):
    # more accurate than sim_seconds
    return float(stats["sim_ticks"]) / float(stats["sim_freq"])

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
            print(f"could not parse line {line}")
            continue
        key = parts[0]
        if isinstance(value, str):
            try:
                value = value.replace("%", "").replace("nan", "None").replace("inf", "None").replace("-inf", "None")
                value = ast.literal_eval(value) if value != "None" else None
            except:
                print(f"could not parse value {value} for key {key}")
        elif isinstance(value, list):
            try:
                value = [v.replace("%", "").replace("nan", "None").replace("inf", "None").replace("-inf", "None") for v in value]
                value = [ast.literal_eval(v) if v!= "None" else None for v in value]
            except:
                print(f"could not parse value {value} for key {key}")
        
        stats[key] = value
    stats["sim_seconds_precise"] = calculate_sim_seconds(stats)
    return stats

def get_average_time(corresponding_benchmark):
    gem5_stats = glob.glob(corresponding_benchmark + "/gem5_stats.*.txt")
    sim_seconds_precise_all = []
    for gem5_stat in gem5_stats:
        try:
            stats = parse_stats_txt(gem5_stat)
            sim_seconds_precise = stats["sim_seconds_precise"]
            sim_seconds_precise_all.append(sim_seconds_precise)
        except:
            continue

    if len(sim_seconds_precise_all) != 0:
        return sum(sim_seconds_precise_all) / len(sim_seconds_precise_all)
    else:
        return 820
    

record = dict()
for each_generate_out in tqdm(generate_outs):
    each_generate = each_generate_out.replace(".out", "")


    corresponding_benchmark = os.path.join(BASE, generated_model_id, generate_project, "benchmark_gem5_testcases_3", each_generate)
    (flag_exec_right_gem5_right, flag_all_right) = result_statistics(corresponding_benchmark)
    average_time = get_average_time(corresponding_benchmark)
    identifier = get_identifier(each_generate)

    item = {
    "can_pass_testcases": flag_exec_right_gem5_right,
    "pass_testcases_and_answer_correct": flag_all_right,
    "average_time": average_time
    }

    if identifier in record:
        record[identifier][each_generate] = item
    else:
        record[identifier] = {}
        record[identifier][each_generate] = item

with open(os.path.join(BASE, generated_model_id, generate_project, "analysis_results.json"), 'w') as writer:
    json.dump(record, writer, indent=4)
