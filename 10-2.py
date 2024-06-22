# pip install pycryptodome
from glob import glob
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection_2"

def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="🤗"
    )
    st.sidebar.title("Nav")
    st.session_state.costs = []

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    else:
        st.session_state.model_name = "gpt-4"

    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)

def get_pdf_text(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
    return text

def fix_encoding(text):
    # 様々なエンコーディングでテキストを試す
    possible_encodings = ['utf-8', 'latin1', 'shift_jis', 'iso-2022-jp']
    for enc in possible_encodings:
        try:
            decoded_text = text.encode('latin1').decode(enc)
            return decoded_text
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return text  # どのエンコーディングでもうまくいかなかった場合は元のテキストを返す

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-ada-002",
        chunk_size=500,
        chunk_overlap=0,
    )
    return text_splitter.split_text(text)

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

def build_vector_store(pdf_text):
    qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)

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

def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")
    container = st.container()
    with container:
        uploaded_file = st.file_uploader(label='Upload your PDF here😇', type='pdf')
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                pdf_text = get_pdf_text(uploaded_file)
                pdf_text = fix_encoding(pdf_text)  # エンコーディングを修正
                pdf_chunks = split_text(pdf_text)
                st.write("Extracted Text:")
                for chunk in pdf_chunks:
                    st.write(chunk)
            with st.spinner("Building vector store..."):
                build_vector_store(pdf_chunks)
            st.success("PDF uploaded and processed successfully!")

def ask(qa, query):
    with get_openai_callback() as cb:
        answer = qa(query)
    return answer, cb.total_cost

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

