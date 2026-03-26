import torch
# Đổi Qwen2_5_VL thành Qwen2VL
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def apply_lora_to_quantized_model(model_path):
    # Cần định nghĩa lại config nén để load model 4-bit đúng cách
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # Dùng đúng class Qwen2VL
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config, # Thêm dòng này để load model đã nén
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True # Tránh khởi tạo lại weights gây lỗi 'Byte'
    )

    # Chuẩn bị model cho k-bit training (LoRA trên model nén)
    model = prepare_model_for_kbit_training(model)

    # Target các layer quan trọng để học reasoning trong ScienceQA
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    lora_config = LoraConfig(
        r=16,                
        lora_alpha=32,       
        target_modules=target_modules,
        # Qwen2-VL dùng "visual" cho phần encoder ảnh
        exclude_modules=["visual"], 
        lora_dropout=0.05,   
        bias="none",         
        task_type="CAUSAL_LM" 
    )

    peft_model = get_peft_model(model, lora_config)

    # Đảm bảo phần Vision không bị train để tiết kiệm VRAM Kaggle
    for name, param in peft_model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    peft_model.print_trainable_parameters()
    
    visual_is_training = any(p.requires_grad for name, p in peft_model.named_parameters() if "visual" in name)
    print(f"--- Vision Encoder Status: {'Training' if visual_is_training else 'Frozen (Safe)'} ---")

    return peft_model