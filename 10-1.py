# 必要なライブラリをインポート
from glob import glob
import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Qdrant の設定
QDRANT_PATH = "./local_qdrant"  # ローカルQdrantのパス
COLLECTION_NAME = "my_collection_2"  # コレクション名

# ページの初期化
def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",  # ページのタイトルを設定
        page_icon="🤗"  # ページのアイコンを設定
    )
    st.sidebar.title("Nav")  # サイドバーのタイトルを設定
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
    return text

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
        uploaded_file = st.file_uploader(label='Upload your PDF here😇', type='pdf')  # ファイルアップローダーを作成
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):  # テキスト抽出中のスピナーを表示
                pdf_text = get_pdf_text(uploaded_file)  # PDFからテキストを抽出
                st.write("Extracted Text:")  # 抽出されたテキストのラベルを表示
                st.write(pdf_text)  # 抽出されたテキストを表示
                pdf_chunks = split_text(pdf_text)  # テキストをチャンクに分割
            with st.spinner("Building vector store..."):  # ベクトルストア構築中のスピナーを表示
                build_vector_store(pdf_chunks)  # ベクトルストアを構築
            st.success("PDF uploaded and processed successfully!")  # 成功メッセージを表示

# 質問を処理する関数
def ask(qa, query):
    with get_openai_callback() as cb:  # OpenAIのコールバックを設定
        answer = qa(query)  # 質問に対する回答を取得
    return answer, cb.total_cost  # 回答とコストを返す

# 質問ページを表示する関数
def page_ask_my_pdf():
    st.title("Ask My PDF(s)")  # ページのタイトルを設定
    llm = select_model()  # モデルを選択
    container = st.container()  # コンテナを作成
    response_container = st.container()  # レスポンスコンテナを作成
    with container:
        query = st.text_input("Query: ", key="input")  # クエリの入力欄を作成
        if not query:
            answer = None  # クエリがない場合は回答をNoneに設定
        else:
            qa = build_qa_model(llm)  # 質問応答モデルを構築
            if qa:
                with st.spinner("ChatGPT is typing ..."):  # ChatGPTの処理中スピナーを表示
                    answer, cost = ask(qa, query)  # 質問に対する回答を取得
                st.session_state.costs.append(cost)  # コストをセッションステートに追加
            else:
                answer = None  # モデルがない場合は回答をNoneに設定
        if answer:
            with response_container:
                st.markdown("## Answer")  # 回答のラベルを表示
                st.write(answer)  # 回答を表示

# メイン関数
def main():
    init_page()  # ページを初期化
    selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])  # サイドバーのラジオボタンでページを選択
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()  # PDFアップロードページを表示
    elif selection == "Ask My PDF(s)":
        page_ask_my_pdf()  # 質問ページを表示
    costs = st.session_state.get('costs', [])  # セッションステートからコストを取得
    st.sidebar.markdown("## Costs")  # サイドバーにコストのラベルを表示
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")  # 合計コストを表示
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")  # 各コストをサイドバーに表示

if __name__ == '__main__':
    main()  # メイン関数を実行

