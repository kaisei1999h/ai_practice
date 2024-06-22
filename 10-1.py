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
QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection_2"

# ページの初期化
def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="🤗"
    )
    st.sidebar.title("Nav")
    st.session_state.costs = []

# モデルの選択
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4", "GPT-4o"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    elif model == "GPT-4":
        st.session_state.model_name = "gpt-4"
    else:
        st.session_state.model_name = "gpt-4o"

    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)

# PDF からテキストを抽出する関数
def get_pdf_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

# テキストをチャンクに分割する関数
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-ada-002",
        chunk_size=500,
        chunk_overlap=0,
    )
    return text_splitter.split_text(text)

# Qdrant をロードする関数
def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=OpenAIEmbeddings()
    )

# ベクトルストアを構築する関数
def build_vector_store(pdf_text):
    qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)

# 質問応答モデルを構築する関数
def build_qa_model(llm):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        search_type="similarity",
        search_kwargs={"k":10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

# PDF アップロードとベクトルストアの構築ページを表示する関数
def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")
    container = st.container()
    with container:
        uploaded_file = st.file_uploader(label='Upload your PDF here😇', type='pdf')
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                pdf_text = get_pdf_text(uploaded_file)
                st.write("Extracted Text:")
                st.write(pdf_text)
                pdf_chunks = split_text(pdf_text)
            with st.spinner("Building vector store..."):
                build_vector_store(pdf_chunks)
            st.success("PDF uploaded and processed successfully!")

# 質問を処理する関数
def ask(qa, query):
    with get_openai_callback() as cb:
        answer = qa(query)
    return answer, cb.total_cost

# 質問ページを表示する関数
def page_ask_my_pdf():
    st.title("Ask My PDF(s)")
    llm = select_model()
    container = st.container()
    response_container = st.container()
    with container:
        query = st.text_input("Query: ", key="input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)
            else:
                answer = None
        if answer:
            with response_container:
                st.markdown("## Answer")
                st.write(answer)

# メイン関数
def main():
    init_page()
    selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Ask My PDF(s)":
        page_ask_my_pdf()
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()

