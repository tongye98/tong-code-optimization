python src/export_model.py \
    --model_name_or_path /home/tongye2/llm_checkpoints/CodeLlama-13b-hf/models--codellama--CodeLlama-13b-hf/snapshots/a49a368460ad22e43dfffb97a1e1b826a6418d3b/ \
    --adapter_name_or_path saves/codellama_13b_sft_pie_cpp/checkpoint-2754/ \
    --template default \
    --finetuning_type lora \
    --export_dir saves/codellama_13b_sft_pie_cpp/checkpoint-2754/full_model