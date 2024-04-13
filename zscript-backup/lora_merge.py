import transformers
import peft

from peft import PeftConfig,PeftModelForCausalLM
from transformers import AutoModelForCausalLM
peft_model_id = "/data/tongye/saves/codellama_13b_sft_pie_cpp_0123/checkpoint-2754/"
config = PeftConfig.from_pretrained(peft_model_id)
print("config = {}".format(config))

model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,use_safetensors=False,torch_dtype="auto")
model = PeftModelForCausalLM.from_pretrained(model, peft_model_id)


model = model.merge_and_unload()

out_dir = "/data/tongye/saves/codellama_13b_sft_pie_cpp_0123/full_model/"
model.save_pretrained(out_dir, safe_serialization=False)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.save_pretrained(out_dir)