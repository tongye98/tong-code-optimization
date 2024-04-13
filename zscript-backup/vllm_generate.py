from vllm import LLM,SamplingParams
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--test_data_pth",type=str,default="data/pie/cpp/test.json")
parser.add_argument("--model_checkpoint",type=str,default="/data/tongye/saves/codellama_13b_sft_pie_cpp_0123/full_model/")
parser.add_argument("--use_beam_search",type=bool,default=False)
parser.add_argument("--n",type=int,default=1)
parser.add_argument("--temperature",type=float,default=1.0)
parser.add_argument("--top_p",type=float,default=1.0)
parser.add_argument("--max_tokens",type=int,default=1024)
parser.add_argument("--result_pth",type=str,default="/data/tongye/saves/codellama_13b_sft_pie_cpp_0123/result_1sample_temp1_topp1.json")

args = parser.parse_args()


llm = LLM(args.model_checkpoint,swap_space=8, gpu_memory_utilization=0.9)
if args.use_beam_search:
    gen_param = SamplingParams(use_beam_search=True,n=args.n,max_tokens=args.max_tokens)
else:
    gen_param = SamplingParams(n=args.n,max_tokens=args.max_tokens,temperature=args.temperature,top_p=args.top_p)

test_data = json.load(open(args.test_data_pth,'r'))
chunk_size = 1024
all_out = []
system = "You are a professional programming expert. \
        Your task is to provide modified complete code with the minimal time complexity and faster execution speed, \
        given a problem description (in HTML format) and the original complete code."
query = "Problem description and original code: {}\nModified code:"

for i in tqdm(range(0,len(test_data),chunk_size)):
    chunks = [data['input'] for data in test_data[i:i+chunk_size]]
    # chunks = list()
    # for data in test_data[i:i+chunk_size]:
    #     chunks.append(data['problem_description'] + "\n" + query.format(data['input']))
    out_seqs = llm.generate(prompts=chunks,sampling_params=gen_param)
    all_out += out_seqs

results = []
for in_data,out_data in zip(test_data,all_out):
    new_data = {}
    new_data['input'] = in_data['input']
    new_data['reference'] = in_data['target']
    new_data['candidates'] = [out.text for out in out_data.outputs]
    results.append(new_data)

json.dump(results,open(args.result_pth,'w'),ensure_ascii=False,indent=2)


    


