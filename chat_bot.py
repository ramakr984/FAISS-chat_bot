import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_classic.chains.question_answering import load_qa_chain





# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to create FAISS vector store
def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)


# Load FAISS vector store
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, embeddings,  
                 allow_dangerous_deserialization=True)
    return vector_store


# Build QA Chain
def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever()
    # Load QA chain for combining documents
    llm = ChatGroq(
        #api_key=st.secrets["GROQ_API_KEY"],
        api_key="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",
        model="llama-3.3-70b-versatile"
    )
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    qa_chain = RetrievalQA(retriever=retriever,combine_documents_chain=qa_chain)
    return qa_chain


# Streamlit App
st.title("RAG Chatbot with FAISS and LLaMA")
st.write("Upload a PDF and ask questions based on its content.")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    pdf_path = f"uploaded/{uploaded_file.name}"
    
    # Check if we need to process this file (e.g., if it's different from the last one or not processed yet)
    # For simplicity, we'll process if the chain is None or if the user uploaded a new file (though file_uploader handles state well)
    # Better: Add a button to process to control when it runs, or check simpler condition.
    # Given the structure, we'll verify if we need to rebuild.
    
    # We use a flag in session state to know if we already processed THIS file
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        os.makedirs("uploaded", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = extract_text_from_pdf(pdf_path)

        st.info("Creating FAISS vector store...")
        create_faiss_vector_store(text)

        st.info("Initializing chatbot...")
        st.session_state.qa_chain = build_qa_chain()
        st.session_state.last_uploaded_file = uploaded_file.name
        st.success("Chatbot is ready!")

if st.session_state.qa_chain:
    question = st.text_input("Ask a question about the uploaded PDF:")
    if question:
        st.info("Querying the document...")
        answer = st.session_state.qa_chain.run(question)
        st.success(f"Answer: {answer}")
        st.session_state.last_uploaded_file = uploaded_file.name
     


