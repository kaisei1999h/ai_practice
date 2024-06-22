# 必要なライブラリをインポート
from glob import glob  # ファイルパターンのマッチングに使用
import streamlit as st  # ウェブアプリケーションフレームワーク
import pdfplumber  # PDFからテキストを抽出するライブラリ
from langchain.text_splitter import RecursiveCharacterTextSplitter  # テキストをチャンクに分割するためのライブラリ
from langchain.vectorstores import Qdrant  # ベクトルストアを構築するライブラリ
from langchain.embeddings.openai import OpenAIEmbeddings  # OpenAIの埋め込みモデルを使用するライブラリ
from langchain.chains import RetrievalQA  # 質問応答モデルを構築するライブラリ
from langchain.chat_models import ChatOpenAI  # OpenAIのチャットモデルを使用するライブラリ
from langchain.llms import OpenAI  # OpenAIの言語モデルを使用するライブラリ
from langchain.callbacks import get_openai_callback  # OpenAIのコールバックを取得するライブラリ
from qdrant_client import QdrantClient  # Qdrantクライアントライブラリ
from qdrant_client.models import Distance, VectorParams  # Qdrantのベクトル設定

# Qdrant の設定
QDRANT_PATH = "./local_qdrant"  # ローカルQdrantのパス
COLLECTION_NAME = "study_materials"  # コレクション名

# ページの初期化
def init_page():
    st.set_page_config(
        page_title="Study Helper",  # ページのタイトルを設定
        page_icon="📘"  # ページのアイコンを設定
    )
    st.sidebar.title("Navigation")  # サイドバーのタイトルを設定
    st.session_state.costs = []  # セッションステートにコストのリストを初期化

# モデルの選択
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4", "GPT-4o"))  # モデル選択のラジオボタン
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"  # GPT-3.5 モデル名を設定
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k"  # GPT-3.5-16k モデル名を設定
    elif model == "GPT-4":
        st.session_state.model_name = "gpt-4"  # GPT-4 モデル名を設定
    else:
        st.session_state.model_name = "gpt-4o"  # GPT-4o モデル名を設定

    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300  # 最大トークン数を設定
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)  # 選択したモデルのインスタンスを返す

# PDF からテキストを抽出する関数
def get_pdf_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ''  # テキストを初期化
        for page in pdf.pages:
            text += page.extract_text() + '\n'  # 各ページからテキストを抽出して結合
    return text  # 抽出されたテキストを返す

# テキストをチャンクに分割する関数
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-ada-002",  # 使用するエンコーダーモデルを指定
        chunk_size=500,  # チャンクのサイズを設定
        chunk_overlap=0,  # チャンクの重複を設定
    )
    return text_splitter.split_text(text)  # テキストをチャンクに分割して返す

# Qdrant をロードする関数
def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)  # Qdrant クライアントを作成
    collections = client.get_collections().collections  # コレクションのリストを取得
    collection_names = [collection.name for collection in collections]  # コレクション名のリストを作成

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,  # コレクション名を指定
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),  # ベクトルの設定を指定
        )
        print('collection created')  # コレクションが作成されたことを表示

    return Qdrant(
        client=client,  # Qdrant クライアントを設定
        collection_name=COLLECTION_NAME,  # コレクション名を設定
        embeddings=OpenAIEmbeddings()  # 使用する埋め込みモデルを設定
    )

# ベクトルストアを構築する関数
def build_vector_store(pdf_text):
    qdrant = load_qdrant()  # Qdrant をロード
    qdrant.add_texts(pdf_text)  # テキストを追加してベクトルストアを構築

# 質問応答モデルを構築する関数
def build_qa_model(llm):
    qdrant = load_qdrant()  # Qdrant をロード
    retriever = qdrant.as_retriever(
        search_type="similarity",  # 類似性検索を設定
        search_kwargs={"k":10}  # 検索パラメータを設定
    )
    return RetrievalQA.from_chain_type(
        llm=llm,  # 使用するLLMを設定
        chain_type="stuff",  # チェーンのタイプを設定
        retriever=retriever,  # レトリバーを設定
        return_source_documents=True,  # ソースドキュメントを返す設定
        verbose=True  # 詳細出力を設定
    )

# PDF アップロードとベクトルストアの構築ページを表示する関数
def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")  # ページのタイトルを設定
    container = st.container()  # コンテナを作成
    with container:
        uploaded_file = st.file_uploader(label='Upload your study PDF here📚', type='pdf')  # ファイルアップローダーを作成
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):  # テキスト抽出中のスピナーを表示
                pdf_text = get_pdf_text(uploaded_file)  # PDFからテキストを抽出
                st.write("Extracted Text:")  # 抽出されたテキストのラベルを表示
                st.write(pdf_text)  # 抽出されたテキストを表示
                pdf_chunks = split_text(pdf_text)  # テキストをチャンクに分割
            with st.spinner("Building vector store..."):  # ベクトルストア構築中のスピナーを表示
                build_vector_store(pdf_chunks)  # ベクトルストアを構築
            st.success("PDF uploaded and processed successfully!")  # 成功メッセージを表示

# 問題を生成し、正誤判定を行う関数
def generate_question_and_check_answer(qa, query, user_answer):
    with get_openai_callback() as cb:  # OpenAIのコールバックを設定
        response = qa(query)  # 質問に対する回答を取得
    correct_answer = response["result"]  # 回答を取得
    is_correct = user_answer.lower() in correct_answer.lower()  # 正誤判定を行う
    return correct_answer, is_correct, cb.total_cost  # 回答、正誤判定結果、コストを返す

# 問題を出題するページを表示する関数
def page_ask_my_pdf():
    st.title("Study Questions")  # ページのタイトルを設定
    llm = select_model()  # モデルを選択
    container = st.container()  # コンテナを作成
    response_container = st.container()  # レスポンスコンテナを作成
    with container:
        query = st.text_input("Enter your question: ", key="input")  # クエリの入力欄を作成
        user_answer = st.text_input("Enter your answer: ", key="answer")  # 解答の入力欄を作成
        if query and user_answer:
            qa = build_qa_model(llm)  # 質問応答モデルを構築
            if qa:
                with st.spinner("Checking your answer..."):  # 回答確認中のスピナーを表示
                    correct_answer, is_correct, cost = generate_question_and_check_answer(qa, query, user_answer)  # 正誤判定を行う
                st.session_state.costs.append(cost)  # コストをセッションステートに追加
                if is_correct:
                    st.success(f"Correct! The answer is: {correct_answer}")  # 正解の場合のメッセージを表示
                else:
                    st.error(f"Incorrect. The correct answer is: {correct_answer}")  # 不正解の場合のメッセージを表示
                st.markdown(f"## Explanation")  # 解説のラベルを表示
                st.write(correct_answer)  # 解説を表示
        else:
            st.write("Please enter a question and your answer.")  # クエリと解答の入力を促すメッセージを表示

# メイン関数
def main():
    init_page()  # ページを初期化
    selection = st.sidebar.radio("Go to", ["PDF Upload", "Study Questions"])  # サイドバーのラジオボタンでページを選択
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()  # PDFアップロードページを表示
    elif selection == "Study Questions":
        page_ask_my_pdf()  # 問題出題ページを表示
    costs = st.session_state.get('costs', [])  # セッションステートからコストを取得
    st.sidebar.markdown("## Costs")  # サイドバーにコストのラベルを表示
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")  # 合計コストを表示
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")  # 各コストをサイドバーに表示

if __name__ == '__main__':
    main()  # メイン関数を実行

