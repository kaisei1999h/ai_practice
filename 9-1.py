# Streamlitモジュールをインポート
import streamlit as st

# OpenAIのコールバックを取得するためのモジュールをインポート
from langchain.callbacks import get_openai_callback

# PDFを読み込むためのモジュールをインポート
from PyPDF2 import PdfReader

# OpenAIの埋め込みを利用するためのモジュールをインポート
from langchain.embeddings.openai import OpenAIEmbeddings

# テキストをチャンクに分割するためのモジュールをインポート
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Qdrantベクトルストアを利用するためのモジュールをインポート
from langchain.vectorstores import Qdrant

# Qdrantクライアントを利用するためのモジュールをインポート
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Qdrantのパスとコレクション名を定義
QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection"

# ページを初期化する関数
def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",  # ページのタイトルを設定
        page_icon="🤗"  # ページのアイコンを設定
    )
    st.sidebar.title("Nav")  # サイドバーのタイトルを設定
    st.session_state.costs = []  # セッションステートにコストのリストを追加

# PDFのテキストを取得する関数
def get_pdf_text():
    uploaded_file = st.file_uploader(
        label='Upload your PDF here😇',  # アップロードラベルを設定
        type='pdf'  # アップロード可能なファイルタイプをPDFに限定
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)  # PDFリーダーを使ってアップロードされたファイルを読み込む
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])  # 各ページのテキストを結合
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.emb_model_name,
            # チャンクサイズはPDFによって調整が必要
            chunk_size=250,  # チャンクサイズを設定
            chunk_overlap=0,  # チャンクのオーバーラップを設定
        )
        return text_splitter.split_text(text)  # テキストをチャンクに分割して返す
    else:
        return None  # ファイルがアップロードされていない場合はNoneを返す

# Qdrantクライアントをロードする関数
def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)  # Qdrantクライアントを初期化

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),  # ベクトルサイズと距離を設定
        )
        print('collection created')  # コレクション作成メッセージを出力

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=OpenAIEmbeddings()  # OpenAIの埋め込みを利用
    )

# ベクトルストアを構築する関数
def build_vector_store(pdf_text):
    qdrant = load_qdrant()  # Qdrantをロード
    qdrant.add_texts(pdf_text)  # テキストをQdrantに追加

    # 以下のようにもできる。この場合は毎回ベクトルDBが初期化される
    # LangChain の Document Loader を利用した場合は `from_documents` にする
    # Qdrant.from_texts(
    #     pdf_text,
    #     OpenAIEmbeddings(),
    #     path="./local_qdrant",
    #     collection_name=COLLECTION_NAME,
    # )

# PDFをアップロードしてベクトルDBを構築するページを表示する関数
def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")  # ページタイトルを設定
    container = st.container()  # コンテナを作成
    with container:
        pdf_text = get_pdf_text()  # PDFのテキストを取得
        if pdf_text:
            with st.spinner("Loading PDF ..."):  # スピナーを表示
                build_vector_store(pdf_text)  # ベクトルストアを構築

# PDFに質問するページを表示する関数
def page_ask_my_pdf():
    st.title("Ask My PDF(s)")  # ページタイトルを設定
    st.write('Under Construction')  # 工事中メッセージを表示

    # 後で実装する

# メイン関数
def main():
    init_page()  # ページを初期化

    selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])  # サイドバーにラジオボタンを設定
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()  # PDFアップロードページを表示
    elif selection == "Ask My PDF(s)":
        page_ask_my_pdf()  # 質問ページを表示

    costs = st.session_state.get('costs', [])  # コストのリストを取得
    st.sidebar.markdown("## Costs")  # コストセクションの見出しを設定
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")  # 合計コストを表示
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")  # 各コストを表示

# メイン関数を呼び出す
if __name__ == '__main__':
    main()

