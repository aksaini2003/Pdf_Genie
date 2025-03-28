
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(docs):
    """Extracts text from multiple PDF files."""
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_txt_chunks(text):
    """Splits text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Converts text chunks into vector embeddings and saves them."""
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
   

    vector_store.save_local('faiss_index')

def get_conversational_chain():
    """Creates a LangChain question-answering chain."""
    prompt_template = '''
    answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say, 
    "answer is not available in the context". Don't provide the wrong answer.

    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    '''
    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    return load_qa_chain(model, chain_type='stuff', prompt=prompt)

def user_input(user_question):
    """Handles user input and fetches answers."""
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
   
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({'input_documents': docs, 'question': user_question}, return_only_outputs=True)
    
    st.write('Reply:', response.get('output_text', 'Error generating response'))

def main():

    st.set_page_config(
    page_title="PDF Genie",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

    st.header("Chat with PDFs")
    
    user_question = st.text_input("Ask a question from the uploaded PDFs")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_txt_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete! You can now ask questions.")

if __name__ == "__main__":
    main()
