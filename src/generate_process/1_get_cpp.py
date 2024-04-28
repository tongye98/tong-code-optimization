import os 
import json 
from tqdm import tqdm 

BASE = "/largespace/tydata/code_optimization/cpp/saved_models/"
generated_model_id = "pie-gem5-by-user-cpp_deepseekcoder-7b_sft_moe-algorithm_0426"
generate_project = "generate_2_samples"
test_dataset_path = "/largespace/tydata/code_optimization/cpp/dataset/fine-grained/test_other.json"

def create_single_cpp_source_code(merge_generates_path, target_project_path):
    with open(merge_generates_path, 'r') as f:
        dataset = json.load(f)
    
    for item in tqdm(dataset, desc="Create single cpp code"):
        maybe_faster_codes = item["maybe_faster"]  # dict
        problem_id = item["problem_id"]
        user_id = item["user_id"]
        slow_submission_id = item["slow_submission_id"]
        fast_submission_id = item["fast_submission_id"]

        for idx, maybe_fast_code in maybe_faster_codes.items():
            cpp_source_path = os.path.join(target_project_path, f"{problem_id}_{user_id}_{slow_submission_id}_{fast_submission_id}_maybe_faster_{idx}.cpp")
            with open(cpp_source_path, "w") as g:
                g.write(maybe_fast_code)


if __name__ == "__main__":
    with open(os.path.join(BASE, generated_model_id, generate_project, "generate_2_samples.json"), 'r') as f:
        generates = json.load(f)
    print(f"There are {len(generates)} generated cpp problems.")

    with open(test_dataset_path, 'r') as g:
        test_dataset = json.load(g)
    print(f"There are {len(test_dataset)} test dataset points.")

    assert len(generates) == len(test_dataset), f"{len(generates)} vs {len(test_dataset)}"

    for generate_item, test_item in zip(generates, test_dataset):
        test_item["maybe_faster"] = {}
        samples = len(generate_item["predict"])
        for idx in range(samples):
            test_item["maybe_faster"][idx] = generate_item["predict"][idx]
    
    merge_generates_path = os.path.join(BASE, generated_model_id, generate_project, "merge_generates.json")
    with open(merge_generates_path, 'w') as writer:
        json.dump(test_dataset, writer, indent=4)
    
    target_project_path = os.path.join(BASE, generated_model_id, generate_project, "generate_cpp")
    if not os.path.isdir(target_project_path):
        os.makedirs(target_project_path)
    create_single_cpp_source_code(merge_generates_path, target_project_path)