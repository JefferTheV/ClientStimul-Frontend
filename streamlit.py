import streamlit as st
import time
import os
from openai import OpenAI
import re
import json
API_INFO_FILE = "/home/yfjin/ClientStimul/run/api_info.txt"

API_KEY = "EMPTY" # vLLM 本地部署通常不需要 key
MODEL_NAME = "client-stimul" # 对应 vllm 启动参数 --served-model-name
def get_api_base():
    """尝试从共享文件中读取 vLLM 的地址"""
    if not os.path.exists(API_INFO_FILE):
        return None
    try:
        with open(API_INFO_FILE, "r") as f:
            url = f.read().strip()
        if url.startswith("http"):
            return url
    except:
        pass
    return None

# --- 页面配置 ---
st.set_page_config(page_title="ClientStimul (Slurm版)", layout="wide")

# --- 检查连接状态 ---
api_base = get_api_base()

if not api_base:
    st.warning("⚠️ 等待 vLLM 服务启动...")
    st.info(f"请确保 Slurm 作业已提交，且正在向 {API_INFO_FILE} 写入地址。")
    if st.button("🔄 刷新状态"):
        st.rerun()
    st.stop() # 停止渲染下方内容，直到获取到 IP

# --- 侧边栏 ---
with st.sidebar:
    st.success(f"✅ 已连接后端: {api_base}")
    


    st.divider()
    
    st.subheader("👤 用户画像 (Persona)")
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

# --- 初始化 OpenAI 客户端 ---
@st.cache_resource
def get_client(base_url):
    return OpenAI(api_key="EMPTY", base_url=base_url)

client = get_client(api_base)

# --- 辅助函数：构建 System Prompt ---
def build_system_prompt(persona_json_str):
    # 这里只构建 System Prompt 的内容，历史记录交给 OpenAI SDK 管理
    return f"""## 角色扮演：客户
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

# --- 辅助函数：解析输出 (保持不变) ---
def parse_response(raw_text):
    thinking = ""
    label = ""
    
    think_match = re.search(r"<thinking>(.*?)</thinking>", raw_text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()

    label_match = re.search(r"<behavior_label>(.*?)</behavior_label>", raw_text, re.DOTALL)
    if label_match:
        label = label_match.group(1).strip()
    
    clean_speech = re.sub(r"<thinking>.*?</thinking>", "", raw_text, flags=re.DOTALL)
    clean_speech = re.sub(r"<behavior_label>.*?</behavior_label>", "", clean_speech, flags=re.DOTALL)
    
    return {
        "thinking": thinking,
        "label": label,
        "speech": clean_speech.strip(),
        "raw": raw_text
    }

# --- 聊天界面 ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # 助手消息需要解析
            parsed = parse_response(msg["content"])
            if parsed["label"]:
                st.caption(f"🏷️ **行为标签:** {parsed['label']}")
            if parsed["thinking"]:
                with st.expander("💭 内心活动 (Thinking)"):
                    try:
                        st.json(json.loads(parsed["thinking"]))
                    except:
                        st.markdown(parsed["thinking"])
            st.markdown(parsed["speech"])

# 处理输入
if prompt := st.chat_input("输入你的咨询话语..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 1. 构建消息列表：System Prompt + 历史消息
        api_messages = [
            {"role": "system", "content": build_system_prompt(persona_input)}
        ]
        
        # 添加历史记录
        for m in st.session_state.messages:
            if m["role"] == "user":
                # 用户消息直接添加
                api_messages.append({"role": "user", "content": m["content"]})
            else:
                # 助手消息：解析并只提取 speech 部分，去除 thinking 和 label
                parsed_hist = parse_response(m["content"])
                # 只有当 speech 不为空时才添加（防止出现空消息报错）
                if parsed_hist["speech"]:
                    api_messages.append({"role": "assistant", "content": parsed_hist["speech"]})

        # 2. 调用 API
        try:
            # 使用 stream=True 可以实现打字机效果，这里为了简化先用非流式，
            # 如果需要流式，解析 JSON 结构会稍微复杂一点（因为标签是一点点出来的）。
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                temperature=1.0,
                max_tokens=512,
                top_p=0.9,
                stop=["USER:", "user", "用户", "<|im_end|>"] # vLLM 通常会自动处理 eos_token
            )
            
            response_text = completion.choices[0].message.content
            
            # 3. 解析并显示
            parsed = parse_response(response_text)
            
            if parsed["label"]:
                st.caption(f"🏷️ **行为标签:** {parsed['label']}")
            
            message_placeholder.markdown(parsed["speech"])
            
            if parsed["thinking"]:
                with st.expander("💭 查看内心活动 (Thinking)", expanded=True):
                    try:
                        st.json(json.loads(parsed["thinking"]))
                    except:
                        st.text(parsed["thinking"])

            # 4. 保存到历史
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            st.error(f"API 调用出错: {e}")