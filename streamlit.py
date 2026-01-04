import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import re

# --- 配置路径 (根据你的 trainfull.py 修改) ---
# 基础模型路径
BASE_MODEL_PATH = "/home/yfjin/ClientStimul/Qwen2.5-7B-Instruct"
# GRPO 训练后的最终模型路径 (trainfull.py 中保存的位置)
ADAPTER_PATH = "/home/yfjin/ClientStimul/trl/CS_grpo_new_fullapi/checkpoint-1800"

# --- 页面配置 ---
st.set_page_config(page_title="ClientStimul 模拟来访者测试", layout="wide")

# --- 侧边栏：配置与画像 ---
with st.sidebar:
    st.title("⚙️ 设置")


    
    st.divider()
    
    st.subheader("生成参数")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.8)
    max_new_tokens = st.slider("Max New Tokens", 128, 1024, 512)
    top_p = st.slider("Top P", 0.0, 1.0, 0.9)

    st.divider()
    
    st.subheader("👤 用户画像 (Persona)")
    # 默认画像来自你的 prompt.py
    default_persona = {
        "background": "中文母语的年轻求职者，曾有工作经历，性格偏内向、敏感于人际拒绝与“丢脸”。近期持续求职未获offer，在宏观就业不景气的背景下触发强烈焦虑与自我怀疑。为提升面试表现曾请面试辅导并完成第一版材料，但在需要进一步挖掘与表述自身经历时动力下降、停滞。过往领导反馈其人际/职场社交需加强，促使其怀疑自己是否适合职场文化。曾线下做过心理咨询。",
        "chief_complaint": "表面诉求是“找不到工作、很焦虑、担心被淘汰”，想提升面试与求职效果、缓解焦虑并恢复动力。更深层的是自我价值感受外界评价强烈影响，内向特质在职场社交情境中带来负回馈，导致回避与自我否定循环；对努力与结果的关系存在悲观预期，因缺少即时正反馈而难以坚持。",
        "cognitive_patterns": "存在灾难化（担心被淘汰）、过度概括（没拿到offer就归因为自身不行）、读心与标签化（社会不喜欢内向、自己“不够优秀/不适合职场”）、选择性注意与贬低自身（看见他人优秀和努力，忽视自身匹配度与已有尝试）、结果预言与外控倾向（结果靠运气，努力也可能无用）、非此即彼/条件式信念（不外向就难成功；没有外界正反馈就难以继续）。核心信念倾向于“我不够好/不被看见”“世界很竞争且苛刻”，条件假设为“若主动社交被拒就很丢脸，说明我不合适”。",
        "emotional_behavioral": "主导情绪：焦虑、自我怀疑、羞耻/尴尬、羡慕他人、无力与挫败；在退出社交、成为旁观者时感到放松与安全。行为上表现为回避（减少主动请教/社交）、拖延（面试材料二稿停滞）、计划-执行脱节（夜晚计划、白天不落实）、社会比较与刷求职App寻同温层、在负反馈时自我打击；同时也有积极求助（辅导、咨询）、愿意尝试自我奖励与小步暴露的倾向。",
        "speech_style": "礼貌、合作且反思性强；能清晰描述内在体验与困惑，逻辑性好，会提出具体问题与类比（奖励机制比喻）；语气中性偏悲观但开放接纳建议，无明显对抗。",
        "resistance_level": "低",
        "strengths_resources": "自省能力强，能识别“与外界一起打击自己”的模式；愿意求助与学习（面试辅导、心理咨询）；能理解并接受小步改变与自我强化的策略；重视真实性与边界，具备稳定的自我安抚方式（独处、旁观者姿态）；观察力与思考深度可转化为职场优势；线上同伴群体可提供规范化参照与支持；已有工作经验可作为可挖掘的能力证据。"
    }
    
    persona_input = st.text_area(
        "编辑 JSON 画像", 
        value=json.dumps(default_persona, indent=2, ensure_ascii=False),
        height=400
    )

    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# --- 加载模型 ---
@st.cache_resource
def load_model(base_path, adapter_path):
    status_text = st.empty()
    status_text.info("正在加载 Tokenizer 和 Base Model...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 根据配置决定是否使用量化
    quantization_config = None
    

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2"  # 你的脚本里用了 flash_attention_2
    )
    
    status_text.info(f"正在加载 LoRA Adapter: {adapter_path} ...")
    # 加载 GRPO 训练后的 Adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    status_text.success("模型加载完成！")
    return model, tokenizer

try:
    model, tokenizer = load_model(BASE_MODEL_PATH, ADAPTER_PATH)
except Exception as e:
    st.error(f"模型加载失败。请检查路径是否正确。\n错误信息: {e}")
    st.stop()

# --- 辅助函数：构建 Prompt ---
def build_prompt_with_history(history, persona_json_str):
    # 这里复制了 prompt.py 中的 System Prompt 结构
    system_prompt_text = f"""## 角色扮演：客户
你正在扮演一个正在接受心理咨询的客户。

## 任务指令
你必须严格遵循 [用户画像] 来回应咨询师（角色为 'user'）的发言。
你的每一次回应都必须严格遵循以下三部分格式：

1.  **<thinking>...</thinking>**: 首先，生成一个 JSON 对象，包含你（客户）的内心活动。
2.  **<behavior_label>...</behavior_label>**: 其次，从下面提供的11个标签中，选择一个最能描述你接下来发言的标签。
3.  **实际发言**: 最后，写下你（客户）实际说出口的话。

## 行为标签 (必须从此列表中选择)：
1.  确认 (Confirming)
2.  提供信息 (Giving Information)
3.  合理请求 (Reasonable Request)
4.  扩展 (Extending)
5.  重构 (Reformulating)
6.  表达困惑 (Expressing Confusion)
7.  防卫 (Defending)
8.  自我批评或绝望 (Self-criticism or Hopelessness)
9.  转移话题 (Shifting Topics)
10. 焦点断开 (Focus Disconnection)
11. 讽刺性回答 (Sarcastic Answer)

## 你的画像 (必须严格遵循)：
{persona_json_str}"""

    # 构建完整的 Prompt 字符串
    full_prompt = f"<|im_start|>system\n{system_prompt_text}<|im_end|>\n"
    
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        content = msg["content"]
        # 注意：历史记录里存的是纯文本，我们需要把 Assistant 的完整输出（含标签）拼回去
        full_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # 添加当前的 User 引导
    full_prompt += "<|im_start|>assistant\n"
    
    return full_prompt

# --- 辅助函数：解析输出 ---
def parse_response(raw_text):
    """
    解析模型输出，提取 thinking, label 和 speech
    """
    thinking = ""
    label = ""
    speech = raw_text

    # 提取 <thinking>
    think_match = re.search(r"<thinking>(.*?)</thinking>", raw_text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        # 从 raw_text 中移除 thinking 部分，方便后续显示
        # speech = speech.replace(think_match.group(0), "")

    # 提取 <behavior_label>
    label_match = re.search(r"<behavior_label>(.*?)</behavior_label>", raw_text, re.DOTALL)
    if label_match:
        label = label_match.group(1).strip()
        # speech = speech.replace(label_match.group(0), "")
    
    # 清理 Speech：移除标签后的剩余文本即为 Speech，但也需要处理可能残留的换行
    # 这里做一个简单的处理：把标签都删掉，剩下的就是 Speech
    clean_speech = re.sub(r"<thinking>.*?</thinking>", "", raw_text, flags=re.DOTALL)
    clean_speech = re.sub(r"<behavior_label>.*?</behavior_label>", "", clean_speech, flags=re.DOTALL)
    
    return {
        "thinking": thinking,
        "label": label,
        "speech": clean_speech.strip(),
        "raw": raw_text # 保存原始输出用于下一次历史拼接
    }

# --- 聊天界面逻辑 ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # 如果是助手，尝试解析并美化显示
            parsed = parse_response(msg["content"])
            if parsed["label"]:
                st.caption(f"🏷️ **行为标签:** {parsed['label']}")
            if parsed["thinking"]:
                with st.expander("💭 内心活动 (Thinking)"):
                    try:
                        # 尝试格式化 JSON 显示
                        think_json = json.loads(parsed["thinking"])
                        st.json(think_json)
                    except:
                        st.markdown(parsed["thinking"])
            st.markdown(parsed["speech"])

# 处理用户输入
if prompt := st.chat_input("输入你的咨询话语..."):
    # 1. 显示用户输入
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 生成回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("模拟来访者正在思考..."):
            # 构建 Prompt
            full_prompt_str = build_prompt_with_history(st.session_state.messages, persona_input)
            
            inputs = tokenizer(full_prompt_str, return_tensors="pt").to(model.device)
            
            # 设置 Stop Tokens (参考你的 trainfull.py)
            stop_words = ["USER:", "user", "USER", "用户", "<|im_end|>"]
            stop_ids = [tokenizer.convert_tokens_to_ids(w) for w in stop_words]
            # 过滤掉 unknown token
            stop_ids = [idx for idx in stop_ids if idx != tokenizer.unk_token_id]
            if tokenizer.eos_token_id not in stop_ids:
                stop_ids.append(tokenizer.eos_token_id)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stop_ids
                )
            
            # 解码
            # 只取新生成的部分
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 解析内容
            parsed = parse_response(response_text)
            
            # 渲染显示
            display_content = ""
            if parsed["label"]:
                display_content += f"**🏷️ 行为标签:** {parsed['label']}\n\n"
            
            message_placeholder.markdown(parsed["speech"]) # 先显示主要的
            
            if parsed["thinking"]:
                with st.expander("💭 查看内心活动 (Thinking)", expanded=True):
                    try:
                        st.json(json.loads(parsed["thinking"]))
                    except:
                        st.text(parsed["thinking"])
            
            # 更新占位符以显示标签 + 文本
            # (Streamlit 的 expander 不能嵌套在 empty() 更新里，所以上面是即时渲染，这里不用再全量覆盖)

    # 3. 保存完整的原始回复到历史记录（以便下一次 Prompt 拼接时包含标签和 thinking）
    st.session_state.messages.append({"role": "assistant", "content": response_text})