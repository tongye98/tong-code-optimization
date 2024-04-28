import os 
import json 
from tqdm import tqdm 
import matplotlib.pyplot as plt 

dataset_path = "data/cpp_splits/"
file_name = "test.jsonl"

def generate_problems_dataset(language):
    target_path = os.path.join(dataset_path, file_name)

    datas = []
    with open(target_path, 'r') as file:
        for line in file:
            data_line = json.loads(line)
            datas.append(data_line)

    print("Length of datas = {}".format(len(datas)))

    problems = dict()  #


    for sample in tqdm(datas, desc="Progressing:"):
        problem_id = sample["problem_id"]

        # check problem_id in problems
        if problem_id not in problems.keys():
            problems[problem_id] = []
        
        user_id = sample["user_id"]
        function_slow = sample["input"]
        function_fast = sample["target"]
        cpu_time_slow = sample["cpu_time_v0"]
        cpu_time_fast = sample["cpu_time_v1"]
        if language == 'python':
            measured_runtime_slow = sample["measured_runtime_v0"]
            measured_runtime_fast = sample["measured_runtime_v1"]
        submission_id_slow = sample["submission_id_v0"]
        submission_id_fast = sample["submission_id_v1"]

        if language == 'python':
            sample_slow = {"user_id": user_id,
                        "submission_id": submission_id_slow,
                        "cpu_time": cpu_time_slow,
                        "measured_time": measured_runtime_slow,
                        "function": function_slow,
                        }
            
            sample_fast = {"user_id": user_id,
                        "submission_id": submission_id_fast,
                        "cpu_time": cpu_time_fast,
                        "measured_time": measured_runtime_fast,
                        "function": function_fast,
                        }
        else:
            sample_slow = {"user_id": user_id,
                        "submission_id": submission_id_slow,
                        "cpu_time": cpu_time_slow,
                        "function": function_slow,
                        }
            
            sample_fast = {"user_id": user_id,
                        "submission_id": submission_id_fast,
                        "cpu_time": cpu_time_fast,
                        "function": function_fast,
                        }
        
        problems[problem_id].append(sample_slow)
        problems[problem_id].append(sample_fast)

    print("Length of problems = {}".format(len(problems)))

    with open(os.path.join(dataset_path, "problems_split_test.json"), 'w') as json_file:
        json.dump(problems, json_file)

def make_plot(problems_count):  
    # print(problem_count)
    problem_ids = list(problems_count.keys())
    counts = list(problems_count.values())


    plt.figure(figsize=(20,15))
    plt.bar(problem_ids, counts)

    plt.title("Problem Counts Histogram")
    plt.xlabel("Problem_id")
    plt.ylabel("Counts")
    plt.yticks(range(0, 400, 10))
    plt.savefig("cpp_test_problem_counts.png")
    plt.show()

    
def statistics(problems):
    problem_count = dict()
    counts = list()
    for key, value in problems.items():
        count = len(value)
        problem_count[key] = count    
        counts.append(count)

    average = sum(counts) / len(counts)
    print("Average: {}".format(average))

    import statistics
    median = statistics.median(counts)
    print("Median: {}".format(median))

    import numpy as np

    Q1 = np.percentile(counts, 25)  # 第一四分位数
    Q2 = np.percentile(counts, 50)  # 第二四分位数，也是中位数
    Q3 = np.percentile(counts, 75)  # 第三四分位数

    print("Q1 (25th percentile):", Q1)
    print("Q2 (50th percentile):", Q2)
    print("Q3 (75th percentile):", Q3)


if __name__ == "__main__":
    # generate_problems_dataset(language)  # python, cpp

    with open(os.path.join(dataset_path, "problems_split_test.json"), 'r') as json_file:
        problems = json.load(json_file)

        problem_count = dict()
        for key, value in problems.items():
            count = len(value)
            problem_count[key] = count

        sorted_problmes_count = dict(sorted(problem_count.items(), key=lambda x:x[1], reverse=True)[1:])
        make_plot(sorted_problmes_count)
        # statistics(problems)
