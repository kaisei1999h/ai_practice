import streamlit as st
import pdfplumber
import random
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

# PDFのテキストからランダムに一部を取得する関数
def get_random_text_sample(pdf_text, length=1000):
    start = random.randint(0, max(0, len(pdf_text) - length))
    return pdf_text[start:start + length]

# GPTに問題作成を依頼する関数
def generate_question_with_gpt(llm, pdf_text, question_type):
    text_sample = get_random_text_sample(pdf_text)
    prompt = f"""以下のテキストの内容に基づいて、{question_type}の一問一答形式の問題を1つ作成してください。
    問題はランダムに選んでください。前回と同じ問題にならないようにしてください。あなたは学校の先生です。ユーザは日本の中学生です。
    答えは別途出力するため、ここでは問題文のみを出力してください。

    テキスト:
    {text_sample}

    問題を出題してください。
    """
    return llm.predict(prompt)

def check_answer_with_gpt(llm, conversation, user_answer):
    prompt = f"""以下の会話の続きとして、ユーザーの回答に対する正誤判定と詳細な解説を提供してください。
    
    これまでの会話:
    {conversation}

    ユーザーの回答: {user_answer}

    正誤判定と解説:
    """
    return llm.predict(prompt)

def generate_next_question(llm, conversation, pdf_text, question_type):
    text_sample = get_random_text_sample(pdf_text)
    prompt = f"""以下のテキストの内容に基づいて、{question_type}の一問一答形式の問題を1つ作成してください。
    問題は一問のみ作成してください。複数答えさせる問題は作成しないでください。回答は一つのみにしてください。
    問題はランダムに選んでください。前回と同じ,似たような問題にならないようにしてください。
    答えは別途出力するため、ここでは問題文のみを出力してください。あなたは学校の先生です。ユーザは日本の中学生です。
    これまでの会話:
    {conversation}
    
    元のテキスト:
    {text_sample}
    
    問題のタイプ: {question_type}
    
    新しい問題を出題してください。
    """
    return llm.predict(prompt)


def main():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.question_number = 1
        st.session_state.conversation = ""
        st.session_state.current_question = ""
        st.session_state.waiting_for_answer = False
        st.session_state.user_answer = ""
    init_page()
    llm = initialize_model()

    if 'conversation' not in st.session_state:
        st.session_state.conversation = ""
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'waiting_for_answer' not in st.session_state:
        st.session_state.waiting_for_answer = False
    if 'user_answer' not in st.session_state:
        st.session_state.user_answer = ""
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'question_type' not in st.session_state:
        st.session_state.question_type = ""

    uploaded_file = st.file_uploader("勉強用PDFをアップロードしてください📚", type='pdf')
    if uploaded_file:
        with st.spinner("PDFからテキストを抽出中..."):
            st.session_state.pdf_text = get_pdf_text(uploaded_file)
        st.success("PDFの抽出が完了しました！")

    st.session_state.question_type = st.text_input("どのような問題を出してほしいですか？（例：単語の意味を問う、文法について質問する）", value=st.session_state.question_type)
    
    if st.session_state.pdf_text and st.session_state.question_type and not st.session_state.current_question:
        st.session_state.current_question = generate_question_with_gpt(llm, st.session_state.pdf_text, st.session_state.question_type)
        st.session_state.conversation += f"\n質問: {st.session_state.current_question}"
        st.session_state.waiting_for_answer = True

    if st.session_state.current_question:
        st.write(st.session_state.current_question)

    if st.session_state.waiting_for_answer:
        user_answer = st.text_input("あなたの回答を入力してください:", key="answer_input", value="")
        if user_answer:
            st.session_state.conversation += f"\nユーザーの回答: {user_answer}"
            feedback = check_answer_with_gpt(llm, st.session_state.conversation, user_answer)
            st.write("評価結果:")
            st.write(feedback)
            st.session_state.conversation += f"\n評価: {feedback}"
            st.session_state.waiting_for_answer = False
            st.session_state.user_answer = user_answer

    if not st.session_state.waiting_for_answer and st.session_state.current_question:
        if st.button("次の問題へ進む"):
            st.session_state.current_question = generate_next_question(llm, st.session_state.conversation, st.session_state.pdf_text, st.session_state.question_type)
            st.session_state.conversation += f"\n質問: {st.session_state.current_question}"
            st.session_state.waiting_for_answer = True
            st.session_state.user_answer = ""
            st.experimental_rerun()

    # リセットボタンの追加
    if st.button("学習をリセット"):
        # PDF データのみ保持
        pdf_text = st.session_state.pdf_text
        
        # セッション状態をリセット
        for key in list(st.session_state.keys()):
            if key != 'pdf_text':
                del st.session_state[key]
        
        # 保持したPDFデータを再設定
        st.session_state.pdf_text = pdf_text
        
        st.success("学習がリセットされました。新しい問題タイプを入力してください。")
        st.experimental_rerun()

    st.write("現在の会話履歴:")
    st.write(st.session_state.conversation)

if __name__ == '__main__':
    main()