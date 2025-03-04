import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import tempfile

# Set API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAZFzsuBfRHX-Q9i5xkOZOC_wWLi9JyLtk"  # Replace with your key

if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key is missing! Set it in the environment before running.")
    st.stop()

# Efficiently extract text from PDFs (Handles large files)
def extract_text_from_pdf(pdf_docs):
    extracted_text = []
    
    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf.read())  # Save to temp location
            temp_path = temp_file.name

        pdf_reader = PdfReader(temp_path)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Avoid empty pages
                extracted_text.append(page_text.strip())

    return "\n".join(extracted_text)  # Join non-empty texts

# Process Text in Chunks
def get_text_chunks(text):
    if not text.strip():  # Check if text is empty
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Create Vector Store
def get_vector_store(text_chunks):
    if not text_chunks:  # Avoid empty input to vector store
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(text_chunks, embeddings)

# Set up Conversational Chain
def get_conversational_chain(vector_store):
    if not vector_store:
        return None
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")  # Higher quota
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm, vector_store.as_retriever(), memory=memory)

# Handle User Input
def user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        st.write("Human:" if i % 2 == 0 else "Bot:", message.content)

# Main Function
def main():
    st.set_page_config(page_title="DocuQuery: AI-Powered PDF Assistant")
    st.header("📄 DocuQuery: AI-Powered PDF Knowledge Assistant")

    user_question = st.text_input("❓ Ask a question from the PDF")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_question and st.session_state.conversation:
        user_input(user_question)

    with st.sidebar:
        st.title("⚙ Settings")
        st.subheader("📂 Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

        if st.button("🚀 Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF before processing.")
            else:
                with st.spinner("Processing... ⏳"):
                    raw_text = extract_text_from_pdf(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    
                    if not text_chunks:
                        st.error("❌ No text found in the uploaded PDFs!")
                        return
                    
                    vector_store = get_vector_store(text_chunks)
                    if not vector_store:
                        st.error("❌ Unable to create a vector store (Check PDF content).")
                        return

                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("✅ PDF Processed Successfully!")

if __name__ == "__main__":
    main()