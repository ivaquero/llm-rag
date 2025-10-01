import json
import os
import time
import uuid  # 用于生成唯一的聊天 ID

import requests
import streamlit as st

from get_related_data import *  # 假设这个模块是存在的，并且 get_related_data 函数可用


# 这个函数假设你已经定义，保持不变
def get_enhanced_prompt(prompt):
    related_data = get_related_data(prompt, db_path="knowledge_base/vector_db", top_k=5)

    return f"""
    你是一个有用的人工智能助手，你擅长根据提供给你的信息回答用户的问题，以下是根据用户提问在知识库中检索到的辅助信息：
    ===辅助信息===
    {related_data}
    ===用户原始问题===
    {prompt}

    请注意，如果辅助信息与用户问题有关，请严格参考辅助信息进行回答。如果无关，直接回答即可。
    """


# 这个函数假设你已经定义，保持不变
def get_aliyun_response(prompt, selected_model, temperature):
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": get_enhanced_prompt(prompt)},
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.session_state.api_key}",
    }

    payload = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,  # 开启流式传输
    }

    try:
        with requests.post(
            url, headers=headers, json=payload, stream=True, timeout=30
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:].decode("utf-8"))
                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]

                            if choice.get("finish_reason") == "stop":
                                break

                            if (
                                "delta" in choice
                                and "content" in choice["delta"]
                                and choice["delta"]["content"]
                            ):
                                yield choice["delta"]["content"]
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        yield f"Error: {e!s}"


st.set_page_config(page_title="RAG 智能问答机器人", layout="wide")

st.title("AI 智能问答助理")

# ----------- API Key 获取逻辑 (与上次修改相同) -----------
if "api_key" not in st.session_state:
    api_key_found = False

    api_key_from_env = os.environ.get("LLM_API_KEY")
    if api_key_from_env:
        st.session_state.api_key = api_key_from_env
        api_key_found = True
    else:
        try:
            api_key_from_secrets = st.secrets.get("LLM_API_KEY")
            if api_key_from_secrets:
                st.session_state.api_key = api_key_from_secrets
                api_key_found = True
        except Exception:
            pass

    if not api_key_found:
        st.session_state.api_key = ""

# ----------- 初始化会话状态来存储聊天历史和当前聊天 ID -----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 存储所有对话的列表，每个对话是 {id, title, messages: [{role, content}]}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None  # 当前正在查看或进行的对话 ID
if "messages" not in st.session_state:
    st.session_state.messages = []  # 当前对话的消息列表

# 用于控制 chat_input 的临时变量
if "chat_input_value" not in st.session_state:
    st.session_state.chat_input_value = ""

# -------------- 侧边栏内容开始 --------------
with st.sidebar:
    st.header("Setting")

    # API Key 输入
    api_key_input = st.text_input(
        "Aliyun API Key", value=st.session_state.api_key, type="password"
    )
    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
    if not api_key_input and st.session_state.api_key:
        st.session_state.api_key = ""

    # 模型选择
    model_options = ["qwen-max", "qwen-plus", "qwen-turbo"]
    selected_model = st.selectbox("Select Model", model_options, index=2)

    # 温度滑块
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1
    )

    # -------------- 新增：常用快速提问词 --------------
    st.markdown("---")
    st.subheader("常用快速提问词")
    preset_questions = [
        "介绍一下 RAG（Retrieval Augmented Generation）技术",
        "什么是向量数据库？有哪些主流产品？",
        "如何评估一个大语言模型的性能？",
        "请解释一下 Transformer 模型的工作原理",
        "LangChain 框架的主要功能是什么？",
    ]

    # 使用按钮来触发快速提问
    for q in preset_questions:
        if st.button(q, key=f"quick_q_{q}"):
            st.session_state.chat_input_value = q  # 将预设问题填充到临时变量
            # 在 Streamlit 中，设置 session_state 值，然后直接在主逻辑中使用它
            # 不需要立即 rerun，因为 chat_input_value 会在下一个 rerun 周期被检测到。
            # 如果不 rerun, 用户需要手动 enter，这失去了快速提问的意义
            st.rerun()  # 触发一次 rerun 来让主区域的 chat_input 捕获到值

    # -------------- 新增：历史问答数据 --------------
    st.markdown("---")
    st.subheader("历史问答")

    # 新建对话按钮
    if st.button("➕ 新建对话", key="new_chat_button"):
        st.session_state.current_chat_id = None  # 清空当前对话 ID
        st.session_state.messages = []  # 清空当前消息
        st.session_state.chat_input_value = ""  # 清空输入框
        st.rerun()

    # 显示历史对话列表
    if len(st.session_state.chat_history) == 0:
        st.info("还没有历史对话。")
    else:
        # 反转列表，让最新对话显示在最上面
        for _, chat in enumerate(reversed(st.session_state.chat_history)):
            display_title = chat.get(
                "title", f"对话 {chat['id'][:8]}"
            )  # 仅显示 ID 前 8 位
            # 使用一个唯一的 key 来区分按钮
            if st.button(
                display_title,
                key=f"chat_hist_{chat['id']}",
                use_container_width=True,
                type="secondary"
                if chat["id"] != st.session_state.current_chat_id
                else "primary",
            ):
                st.session_state.current_chat_id = chat["id"]
                st.session_state.messages = chat["messages"]  # 加载历史消息到当前对话
                st.session_state.chat_input_value = ""  # 加载历史时清空输入框
                st.rerun()
            # 可以添加一个删除按钮，我暂时注释掉以保持代码简洁，如果你需要可以取消注释
            # if st.button("🗑️", key=f"delete_chat_{chat['id']}", help="删除此对话"):
            #     st.session_state.chat_history = [c for c in st.session_state.chat_history if c['id'] != chat['id']]
            #     if st.session_state.current_chat_id == chat['id']:
            #         st.session_state.current_chat_id = None
            #         st.session_state.messages = []
            #     st.rerun()

    # 清除所有对话按钮
    if st.button("⚠️ 清除所有历史对话", key="clear_all_chats_button"):
        st.session_state.chat_history = []
        st.session_state.current_chat_id = None
        st.session_state.messages = []
        st.session_state.chat_input_value = ""  # 清空输入框
        st.rerun()

# -------------- 侧边栏内容结束 --------------


# ----------- 主界面聊天逻辑 -----------

# 在显示当前消息之前，先检查是否有要预填充的 chat_input_value
if st.session_state.chat_input_value:
    # 如果有值，模拟用户提交这个值
    # 注意：st.chat_input() 的返回是字符串，这里我们直接将其当作 prompt
    # 并且立即清空 chat_input_value，以免下次页面刷新时重复提交
    prompt_from_preset = st.session_state.chat_input_value
    st.session_state.chat_input_value = ""  # 立即清空
    # 将这个来自预设的 prompt 传递给处理函数
    # 这样可以模拟用户输入并立即触发处理逻辑
    # 确保 api_key 在调用前已检查
    if st.session_state.api_key:
        # 执行与用户手动输入相同的逻辑
        st.session_state.messages.append(
            {"role": "Human >>:", "content": prompt_from_preset}
        )

        with st.chat_message("Human >>:"):
            st.markdown(prompt_from_preset)

        with st.chat_message("AI >>:"):
            response_placeholder = st.empty()
            full_response = ""
            with st.spinner("AI 正在思考中..."):
                for response_chunk in get_aliyun_response(
                    prompt_from_preset, selected_model, temperature
                ):
                    full_response += response_chunk
                    response_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)

            response_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "AI >>:", "content": full_response}
            )

            # 保存或更新聊天历史
            if st.session_state.current_chat_id is None:
                new_chat_id = str(uuid.uuid4())  # 使用 UUID 生成唯一 ID
                st.session_state.current_chat_id = new_chat_id
                chat_title = (
                    prompt_from_preset[:30] + "..."
                    if len(prompt_from_preset) > 30
                    else prompt_from_preset
                )
                st.session_state.chat_history.append(
                    {
                        "id": new_chat_id,
                        "title": chat_title,
                        "messages": st.session_state.messages[:],  # 复制当前消息列表
                    }
                )
            else:
                for chat in st.session_state.chat_history:
                    if chat["id"] == st.session_state.current_chat_id:
                        chat["messages"] = st.session_state.messages[:]
                        break
            st.rerun()  # 重新运行以更新侧边栏的历史列表高亮和展示

    else:
        st.warning("请在侧边栏输入您的阿里云 API Key 以开始聊天。")

# 显示当前对话的消息 (在处理了预设问题之后)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat_input 组件本身不带 value 参数。
# 用户手动输入的内容会直接通过 prompt 变量获取。
prompt = st.chat_input(
    "Human Input",
    disabled=not st.session_state.api_key,
    key="user_chat_input",  # 必须有一个唯一的 key
    # 这里不能设置 value 参数，因为它不存在
)

if prompt:
    if not st.session_state.api_key:
        st.warning("请在侧边栏输入您的阿里云 API Key 以开始聊天。")
        st.stop()

    st.session_state.messages.append({"role": "Human >>:", "content": prompt})

    with st.chat_message("Human >>:"):
        st.markdown(prompt)

    with st.chat_message("AI >>:"):
        response_placeholder = st.empty()
        full_response = ""

        with st.spinner("AI 正在思考中..."):
            for response_chunk in get_aliyun_response(
                prompt, selected_model, temperature
            ):
                full_response += response_chunk
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "AI >>:", "content": full_response})

        if st.session_state.current_chat_id is None:
            new_chat_id = str(uuid.uuid4())
            st.session_state.current_chat_id = new_chat_id
            chat_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            st.session_state.chat_history.append(
                {
                    "id": new_chat_id,
                    "title": chat_title,
                    "messages": st.session_state.messages[:],
                }
            )
        else:
            for chat in st.session_state.chat_history:
                if chat["id"] == st.session_state.current_chat_id:
                    chat["messages"] = st.session_state.messages[:]
                    break
        st.rerun()
