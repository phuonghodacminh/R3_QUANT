import torch
import sys
import os
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lora_setup import apply_lora_to_quantized_model
from src.rewards import format_reward_func, accuracy_reward_func
from src.utils import prepare_scienceqa_for_grpo 

def train_r3_quant_grpo(model_dir: str, train_data, output_dir: str):

    processor = AutoProcessor.from_pretrained(model_dir)

    peft_model = apply_lora_to_quantized_model(model_dir)
    
    grpo_dataset = prepare_scienceqa_for_grpo(train_data)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        logging_steps=1,           
        max_steps=500,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8,
        gradient_checkpointing=True, 
        num_generations=4,         
        # SỬA Ở ĐÂY: Sử dụng đúng tên tham số từ danh sách bạn gửi
        max_completion_length=512, # Đã có trong list của bạn
        # Tham số max_prompt_length không có, bạn có thể kiểm soát thông qua 
        # việc truncate dữ liệu đầu vào hoặc để mặc định.
        
        bf16=True,                                  
        remove_unused_columns=False, 
        report_to="none",
        use_cache=False # Tắt use_cache khi dùng gradient checkpointing
    )
    reward_funcs = [
        format_reward_func,
        accuracy_reward_func
    ]

    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=grpo_dataset,
    )

    trainer.train()
    
    print(f"\nĐang lưu mô hình LoRA tại: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir) 

if __name__ == "__main__":
    raw_scienceqa = load_dataset("derek-thomas/ScienceQA", split="validation")
    
    MODEL_DIR = r"./weights/Qwen2-VL-2B-Instruct-4bit"
    OUTPUT_DIR = r"./r3_quant_checkpoints"
    
    train_r3_quant_grpo(MODEL_DIR, raw_scienceqa, OUTPUT_DIR)