import os 
import json


def merge():
    improve_methods_path = "/largespace/tydata/code_optimization/cpp/dataset/fine-grained/"
    with open(os.path.join(improve_methods_path, "response_result_test.json"), 'r') as reader:
        improve_methods = json.load(reader)

    print(len(improve_methods))

    original_path = "/largespace/tydata/code_optimization/cpp/dataset/by_user/test_out_pair_in_original_description.json"
    with open(original_path, 'r') as reader:
        original_dataset = json.load(reader)
        original_dataset = original_dataset

    print(len(original_dataset))

    for data, improve_method in zip(original_dataset, improve_methods):
        assert data["problem_id"] == improve_method["problem_id"]
        assert data["user_id"] == improve_method["user_id"]
        assert data["slow_submission_id"] == improve_method["slow_submission_id"]
        assert data["fast_submission_id"] == improve_method["fast_submission_id"]

        data["improve_method"] = improve_method["improve_method"]

    with open(os.path.join(improve_methods_path, "merge_test.json"), 'w') as writer:
        json.dump(original_dataset, writer, indent=4)

def judge(improve_method):
    if ("Algorithmic changes") in improve_method:
        return True
    else:
        return False

def classify():
    improve_methods_path = "/largespace/tydata/code_optimization/cpp/dataset/fine-grained/"
    with open(os.path.join(improve_methods_path, "merge_test.json"), 'r') as reader:
        dataset = json.load(reader)
    
    print(len(dataset))
    algorithmic = []
    other = []

    for data in dataset:
        improve_method = data["improve_method"]
        Is_algorithmic = judge(improve_method)
        if Is_algorithmic:
            algorithmic.append(data)
        else:
            other.append(data)

    with open(os.path.join(improve_methods_path, "test_algorithm.json"), 'w') as writer:
        json.dump(algorithmic, writer, indent=4)

    with open(os.path.join(improve_methods_path, "test_other.json"), 'w') as writer:
        json.dump(other, writer, indent=4) 

def combine():
    improve_methods_path = "/largespace/tydata/code_optimization/cpp/dataset/fine-grained/"
    with open(os.path.join(improve_methods_path, "test_algorithm.json"), 'r') as reader1:
        test_algorithm = json.load(reader1)

    
    with open(os.path.join("/largespace/tydata/code_optimization/cpp/saved_models/pie-gem5-by-user-cpp_deepseekcoder-7b_sft_moe-algorithm_0426/generate_2_samples", 'r')) as reader2:
        generate_algorithm = json.load(reader2)

    assert len(test_algorithm) == len(generate_algorithm)

if __name__ == "__main__":
    # merge()
    
    # classify()
    combine()