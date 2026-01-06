import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 配置路径 ---
base_model_path = "/home/yfjin/ClientStimul/Qwen2.5-7B-Instruct"
adapter_path = "/home/yfjin/ClientStimul/trl/CS_grpo_new_fullapi/checkpoint-1800"
save_path = "/home/yfjin/ClientStimul/Qwen2.5-7B-ClientStimul-Merged" # 合并后的保存路径

print(f"正在加载基础模型: {base_model_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"正在加载 LoRA: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("正在合并权重...")
model = model.merge_and_unload()

print("正在保存 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

print(f"正在保存合并后的模型到: {save_path}")
model.save_pretrained(save_path)
print("完成！")