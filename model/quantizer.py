import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import os

class Qwen4BitQuantizer:
    def __init__(self, base_model_path, save_path):
        self.base_model_path = base_model_path
        self.save_path = save_path

    def quantize_and_save(self):
        # Cấu hình nén 4-bit NF4 - Rất tốt cho Vision Language Model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"--- Đang nén Qwen2-VL từ: {self.base_model_path} ---")
        
        # Load model trực tiếp vào 4-bit
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

        # Lưu model (Lưu ý: Với BnB, nó sẽ lưu checkpoint dạng nén)
        os.makedirs(self.save_path, exist_ok=True)
        model.save_pretrained(self.save_path)
        
        processor = AutoProcessor.from_pretrained(self.base_model_path)
        processor.save_pretrained(self.save_path)
        print(f"--- Hoàn tất! Model nén lưu tại: {self.save_path} ---")

if __name__ == "__main__":
    BASE_MODEL = "./weights/Qwen2-VL-2B-Instruct"
    SAVE_DIR = "./weights/Qwen2-VL-2B-Instruct-4bit"
    
    quantizer = Qwen4BitQuantizer(BASE_MODEL, SAVE_DIR)
    quantizer.quantize_and_save()