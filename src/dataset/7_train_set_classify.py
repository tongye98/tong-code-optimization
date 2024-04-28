from vllm import LLM,SamplingParams
import argparse
import json
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--train_data_path",type=str,default="/largespace/tydata/code_optimization/cpp/dataset/by_user/train_out_pair_improvement10_description.json")
parser.add_argument("--model_checkpoint",type=str,default="/largespace/tydata/models-hf/CodeLlama-13b-Instruct-hf/")
parser.add_argument("--use_beam_search",type=bool,default=False)
parser.add_argument("--n",type=int,default=1)
parser.add_argument("--temperature",type=float,default=0.7)
parser.add_argument("--top_p",type=float,default=1.0)
parser.add_argument("--max_tokens",type=int,default=2048)
parser.add_argument("--result_pth",type=str,default="/largespace/tydata/code_optimization/cpp/dataset/xxx")

args = parser.parse_args()

def get_prompt(data):
    """
    prepare the data for LLMs.
    """
    system = "You're an expert in improving code efficiency. Now, I'll provide you with the pair of slow and fast codes. Based on the slow and fast code pairs, your response only needs to tell me which category of improvement strategy for the fast code compared to the slow code belongs to one of the following five categories: {'Algorithmic changes', 'Input/Output operations', 'Data Structure modifications', 'Miscellaneous adjustments', 'Other'}. Remember, do not response any other information"
    prompt = system + "\n#Slow code:\n" + data["slow_code"] + "\n#Fast code:\n" + data["fast_code"]
    return prompt

def store(outputs, i, partial_dataset):
    generated_texts = []
    for output, point in zip(outputs, partial_dataset):
        generated_text = output.outputs[0].text
        item = {}
        item["problem_id"] =  point["problem_id"]
        item["user_id"] = point["user_id"]
        item["slow_submission_id"] = point["slow_submission_id"]
        item["fast_submission_id"] = point["fast_submission_id"]
        item["improve_method"] = generated_text
        generated_texts.append(item)
    
    with open(f"/largespace/tydata/code_optimization/cpp/dataset/by_user/improve_methods/train_{i}.json", 'w') as f:
        json.dump(generated_texts, f, indent=4)
    return None 

llm = LLM(args.model_checkpoint,swap_space=128, gpu_memory_utilization=0.9)

if args.use_beam_search:
    gen_param = SamplingParams(use_beam_search=True, n=args.n, max_tokens=args.max_tokens) # beam search
else:
    gen_param = SamplingParams(n=args.n, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)

with open(os.path.join(args.train_data_path), "r") as reader:
    train_dataset = json.load(reader)
    train_dataset = train_dataset

print(f"There are {len(train_dataset)} items in train dataset.")
chunk_size = 1000


# for i in tqdm(range(0,len(train_dataset), chunk_size)):
#     chunks = [get_prompt(data) for data in train_dataset[i:i+chunk_size]]
#     outputs = llm.generate(prompts=chunks, sampling_params=gen_param)

for i in tqdm(range(0, len(train_dataset), chunk_size)):
    chunks = [get_prompt(data) for data in train_dataset[i:i+chunk_size]]
    outputs = llm.generate(prompts=chunks, sampling_params=gen_param)
    store(outputs, i, train_dataset[i:i+chunk_size])


