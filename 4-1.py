import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.callbacks import get_openai_callback


def init_page():
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤗"
    )
    st.header("My Great ChatGPT 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []
        st.session_state.total_tokens = []
        st.session_state.prompt_tokens = []
        st.session_state.completion_tokens = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.01とする
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    return ChatOpenAI(temperature=temperature, model_name=model_name)


def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost, cb.total_tokens, cb.prompt_tokens, cb.completion_tokens


def main():
    init_page()

    llm = select_model()
    init_messages()

    # ユーザーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing ..."):
            answer, cost, total_tokens, prompt_tokens, completion_tokens = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)
        st.session_state.total_tokens.append(total_tokens)
        st.session_state.prompt_tokens.append(prompt_tokens)
        st.session_state.completion_tokens.append(completion_tokens)

    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")

    costs = st.session_state.get('costs', [])
    total_tokens = st.session_state.get('total_tokens', [])
    prompt_tokens = st.session_state.get('prompt_tokens', [])
    completion_tokens = st.session_state.get('completion_tokens', [])

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for i in range(len(costs)):
        st.sidebar.markdown(f"- **Request {i + 1}:**")
        st.sidebar.markdown(f"  - **Total Cost:** ${costs[i]:.5f}")
        st.sidebar.markdown(f"  - **Total Tokens:** {total_tokens[i]}")
        st.sidebar.markdown(f"  - **Prompt Tokens:** {prompt_tokens[i]}")
        st.sidebar.markdown(f"  - **Completion Tokens:** {completion_tokens[i]}")

if __name__ == '__main__':
    main()

