import torch
from datasets import Dataset

def build_scienceqa_prompt(question: str, choices: list) -> str:
    prompt = f"{question}\n\nChoices:\n"
    labels = ["A", "B", "C", "D", "E"]
    
    if not choices:
        prompt += (
            "\nThink step by step and reason based on the image. "
            "Enclose your reasoning process within <think> </think> tags "
            "and provide your FINAL ANSWER within <answer> </answer> tags."
        )
        return prompt

    for i, choice in enumerate(choices):
        prompt += f"{labels[i]}. {choice}\n"
        
    valid_labels = labels[:len(choices)]
    if len(valid_labels) > 1:
        label_str = ", ".join(valid_labels[:-1]) + f" or {valid_labels[-1]}"
    else:
        label_str = valid_labels[0]
        
    prompt += (
        "\nThink step by step and reason based on the image. "
        "Enclose your reasoning process within <think> </think> tags "
        f"and provide your FINAL ANSWER (strictly write 1 letter: {label_str}) within <answer> </answer> tags."
    )
    return prompt

def prepare_scienceqa_for_grpo(raw_dataset, max_samples=None):
    formatted_data = {
        "prompt": [],    
        "answer": [], 
    }
    
    labels = ["A", "B", "C", "D", "E"]
    count = 0 
    
    for item in raw_dataset:
        if max_samples and count >= max_samples:
            break
            
        if item["image"] is None:
            continue
            
        # --- FIX LỖI TẠI ĐÂY: Chuyển Dict sang PIL Image ---
        img_data = item["image"]
        try:
            if isinstance(img_data, dict) and "bytes" in img_data:
                # Giải mã từ bytes
                image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            elif isinstance(img_data, Image.Image):
                image = img_data.convert("RGB")
            else:
                continue 
        except Exception as e:
            print(f"Lỗi ảnh tại dòng {count}: {e}")
            continue
        # -----------------------------------------------

        text_prompt = (
            f"Question: {item['question']}\n\nChoices: {item['choices']}\n\n"
            "Reason step by step. Enclose your reasoning within <think> </think> tags "
            "and your final answer (A/B/C/D) within <answer> </answer> tags."
        )
        
        # Cấu hình messages chuẩn cho Multimodal GRPO
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}, # Bây giờ 'image' đã là PIL Object
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        correct_letter = labels[item["answer"]]
        
        formatted_data["prompt"].append(messages)
        formatted_data["answer"].append(correct_letter)
        count += 1 
        
    return Dataset.from_dict(formatted_data)

def prepare_scienceqa_for_sft(raw_dataset, max_samples=None):
    """
    Format dataset cho SFT: Tách biệt 'messages' và 'images' theo chuẩn thư viện TRL.
    """
    # 1. THÊM CỘT "images" riêng biệt
    formatted_data = {
        "messages": [], 
        "images": [] 
    }
    
    labels = ["A", "B", "C", "D", "E"]
    count = 0 
    
    for item in raw_dataset:
        if max_samples and count >= max_samples:
            break
            
        if item["image"] is None:
            continue
            
        # 2. Câu hỏi của User (Chỉ để type="image" làm placeholder)
        text_prompt = build_scienceqa_prompt(item["question"], item["choices"])
        user_message = {
            "role": "user",
            "content": [
                {"type": "image"}, # QUAN TRỌNG: Xóa "image": item["image"] đi
                {"type": "text", "text": text_prompt}
            ]
        }
        
        # 3. Câu trả lời của Assistant
        correct_letter = labels[item["answer"]]
        solution_text = item.get("solution", "Reasoning based on the image.")
        
        assistant_text = (
            f"<think>\n{solution_text}\n</think>\n"
            f"<answer>{correct_letter}</answer>"
        )
        assistant_message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text}
            ]
        }
        
        # 4. Gom dữ liệu vào 2 cột riêng biệt
        formatted_data["messages"].append([user_message, assistant_message])
        # QUAN TRỌNG: Cột images phải là một mảng (list) chứa các ảnh của hội thoại đó
        formatted_data["images"].append([item["image"]]) 
        count += 1 
        
    return Dataset.from_dict(formatted_data)