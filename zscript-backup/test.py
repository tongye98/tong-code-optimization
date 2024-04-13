import torch 

import transformers
from transformers import AutoModelForCausalLM

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print("Available GPU numble = {}".format(num_devices))

    for i in range(num_devices):
        device = torch.cuda.get_device_properties(i)
        print('', device.name)
        # print("", device.major, device.minor)

model_name = "/data1/llm_checkpoints/CodeLlama-13b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0', torch_dtype=torch.float16)

total_parameters = model.num_parameters()
print("Total parameters in the model: ", total_parameters)


memory_allocated = torch.cuda.memory_allocated(device="cuda:0")
memory_allocated_gb = memory_allocated / (1024**3)

print("Memory allocated by the model in GB:", memory_allocated_gb)


model2 = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:1", torch_dtype=torch.float32)
memory_allocated = torch.cuda.memory_allocated(device="cuda:1")
memory_allocated_gb = memory_allocated / (1024**3)
print("Memory allocated by the model2 in GB: ", memory_allocated_gb)

model3 = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:2")
memory_allocated = torch.cuda.memory_allocated(device="cuda:2")
memory_allocated_gb = memory_allocated / (1024**3)
print("Memory allocated by the model3 in GB: ", memory_allocated_gb)
for param in model3.parameters():
    print(param.dtype)
    break


