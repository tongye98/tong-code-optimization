CUDA_VISIBLE_DEVICES=0,1

deepspeed --num_gpus 2 --master_port=9903  src/train_bash.py \
    --deepspeed zscript/ds_zero2.json \
    --stage sft \
    --do_train \
    --model_name_or_path /data1/llm_checkpoints/CodeLlama-13b-hf/ \
    --dataset pie_python \
    --template vanilla \
    --preprocessing_num_workers 8 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 8 \
    --output_dir /data1/tydata1/saves/codellama_13b_pie_python_sft_0310 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy epoch \
    --evaluation_strategy no \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --overwrite_output_dir \
    --fp16 \
    --print_param_status