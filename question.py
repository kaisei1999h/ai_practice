import streamlit as st
import pdfplumber
from langchain.chat_models import ChatOpenAI

# ページの初期化
def init_page():
    st.set_page_config(page_title="学習ヘルパー", page_icon="📘")
    st.title("PDF 学習支援 (GPT-4)")

# GPT-4モデルの初期化
def initialize_model():
    return ChatOpenAI(temperature=0, model_name="gpt-4o")

# PDF からテキストを抽出する関数
def get_pdf_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages)

# GPTに問題作成を依頼する関数
def generate_question_with_gpt(llm, pdf_text, question_type):
    prompt = f"""以下のテキストの内容に基づいて、一問一答形式の問題を作成してください。
    新しい問題を始める際は、「次の問題」と表示してから問題を提示してください。
    問題は一問のみ作成してください。複数答えさせる問題は作成しないでください。


    テキスト:
    {pdf_text}

    問題のタイプ: {question_type}

    最初の問題を出題してください。
    """
    return llm.predict(prompt)

# GPTに正誤判定を依頼する関数
def check_answer_with_gpt(llm, conversation, user_answer):
    prompt = f"""以下の会話の続きとして、ユーザーの回答に対する正誤判定と詳細な解説を提供してください。
    次の問題は出さずに、正誤判定と解説のみを行ってください。

    これまでの会話:
    {conversation}

    ユーザーの回答: {user_answer}

    正誤判定と解説:
    """
    return llm.predict(prompt)

# 次の問題を生成する関数
def generate_next_question(llm, conversation):
    prompt = f"""以下の会話の続きとして、新しい問題を生成してください。
    「次の問題」と表示してから問題を提示してください。

    これまでの会話:
    {conversation}

    次の問題:
    """
    return llm.predict(prompt)

def main():
    init_page()  # ページを初期化
    llm = initialize_model()  # GPT-4モデルを初期化

    # セッション状態を初期化
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ""
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'waiting_for_answer' not in st.session_state:
        st.session_state.waiting_for_answer = False
    if 'user_answer' not in st.session_state:
        st.session_state.user_answer = ""

    uploaded_file = st.file_uploader("勉強用PDFをアップロードしてください📚", type='pdf')
    if uploaded_file:
        with st.spinner("PDFからテキストを抽出中..."):
            pdf_text = get_pdf_text(uploaded_file)
        st.success("PDFの抽出が完了しました！")

        question_type = st.text_input("どのような問題を出してほしいですか？（例：単語の意味を問う、文法について質問する）")
        if question_type and not st.session_state.current_question:
            st.session_state.current_question = generate_question_with_gpt(llm, pdf_text, question_type)
            st.session_state.conversation += f"\n{st.session_state.current_question}"
            st.session_state.waiting_for_answer = True

        if st.session_state.current_question:
            st.write("現在の問題:")
            st.write(st.session_state.current_question)

        if st.session_state.waiting_for_answer:
            user_answer = st.text_input("あなたの回答を入力してください（終了する場合は「終了」と入力）:", key="answer_input")
            if user_answer.lower() == "終了":
                st.write("学習セッションを終了します。お疲れ様でした！")
                st.session_state.conversation = ""
                st.session_state.current_question = ""
                st.session_state.waiting_for_answer = False
                st.session_state.user_answer = ""
            elif user_answer:
                st.session_state.conversation += f"\nユーザーの回答: {user_answer}"
                feedback = check_answer_with_gpt(llm, st.session_state.conversation, user_answer)
                st.write("評価結果:")
                st.write(feedback)
                st.session_state.conversation += f"\n{feedback}"
                st.session_state.waiting_for_answer = False
                st.session_state.user_answer = user_answer

        if not st.session_state.waiting_for_answer and st.session_state.current_question:
            if st.button("次の問題へ進む"):
                st.session_state.current_question = generate_next_question(llm, st.session_state.conversation)
                st.session_state.conversation += f"\n{st.session_state.current_question}"
                st.session_state.waiting_for_answer = True
                st.session_state.user_answer = ""
                st.experimental_rerun()  # Streamlitの再実行をトリガー

        st.write("現在の会話履歴:")
        st.write(st.session_state.conversation)

if __name__ == '__main__':
    main()