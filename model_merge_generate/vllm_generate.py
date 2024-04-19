from vllm import LLM,SamplingParams
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--test_data_pth",type=str,default="/largespace/tydata/code_optimization/cpp/by_user/test_out_pair_in_original.json")
parser.add_argument("--model_checkpoint",type=str,default="/largespace/tydata/code_optimization/cpp/saved_models/pie-gem5-by-user-cpp_deepseekcoder-7b_sft_0418/full_model/")
parser.add_argument("--use_beam_search",type=bool,default=False)
parser.add_argument("--n",type=int,default=1)
parser.add_argument("--temperature",type=float,default=0.7)
parser.add_argument("--top_p",type=float,default=1.0)
parser.add_argument("--max_tokens",type=int,default=1024)
parser.add_argument("--result_pth",type=str,default="/largespace/tydata/code_optimization/cpp/saved_models/pie-gem5-by-user-cpp_deepseekcoder-7b_sft_0418/generate/result_1sample_temp07_topp1.json")

args = parser.parse_args()


llm = LLM(args.model_checkpoint,swap_space=64, gpu_memory_utilization=0.9)
if args.use_beam_search:
    gen_param = SamplingParams(use_beam_search=True, n=args.n, max_tokens=args.max_tokens) # beam search
else:
    gen_param = SamplingParams(n=args.n, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)

test_data = json.load(open(args.test_data_pth,'r'))
chunk_size = 1024
all_out = []

for i in tqdm(range(0,len(test_data), chunk_size)):
    chunks = [data['slow_code'] for data in test_data[i:i+chunk_size]]
    out_seqs = llm.generate(prompts=chunks, sampling_params=gen_param)
    all_out += out_seqs

results = []
for in_data, out_data in zip(test_data, all_out):
    new_data = {}
    new_data['problem_id'] = in_data['problem_id']
    new_data['slow_user_id'] = in_data['user_id']
    new_data["slow_submission_id"] = in_data['slow_submission_id']
    new_data["slow_time"] = in_data['slow_time']
    new_data['input_slow_code'] = in_data['slow_code']
    new_data['fast_user_id'] = in_data['user_id']
    new_data['fast_submission_id'] = in_data['fast_submission_id']
    new_data['reference_fast_code'] = in_data['fast_code']
    new_data['fast_time'] = in_data['fast_time']
    new_data['candidates_maybe_faster_code'] = [out.text for out in out_data.outputs]
    results.append(new_data)

json.dump(results, open(args.result_pth, 'w'), ensure_ascii=False, indent=4)
