import os 
import json 

BASE = "/largespace/tydata/code_optimization/cpp/saved_models/"
generated_model_id = "pie-gem5-by-user-cpp_deepseekcoder-7b_sft_moe-algorithm_0426"
generate_project = "generate_2_samples"

with open(os.path.join(BASE, generated_model_id, generate_project, "analysis_results.json"), 'r') as reader:
    overall_result = json.load(reader)

print(f"There are {len(overall_result)} unique test problem answer.")

recode_fast = dict()
for test_problem_identifier, solutions in overall_result.items():
    solutions_count = len(solutions)
    pass_testcases_solution = []

    for current_solution, current_solution_result in solutions.items():
        can_pass_testcases = current_solution_result["can_pass_testcases"]
        if can_pass_testcases:
            pass_testcases_solution.append(current_solution)
    
    fast = None
    if len(pass_testcases_solution) == 1:
        fast = pass_testcases_solution[0]
    elif len(pass_testcases_solution) > 1:
        time_rank = {}
        for candidata in pass_testcases_solution:
            time_rank[candidata] = solutions[candidata]["average_time"]
        min_pair = min((time, candidata) for candidata, time in time_rank.items())
        fast = min_pair[1]
    else:
        fast = None

    if fast is not None:
        recode_fast[test_problem_identifier] = {fast: solutions[fast]}

with open(os.path.join(BASE, generated_model_id, generate_project, "fast_solutions.json"), 'w') as writer:
    json.dump(recode_fast, writer, indent=4)