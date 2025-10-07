


# def main():
#     #in this we are going to made this application using the langchain and streamlit

#     #by langchain we can create this application by using the google api 
#     import streamlit as st 
#     from PyPDF2 import PdfReader
#     from langchain.text_splitter import RecursiveCharacterTextSplitter 
#     import os 

#     #as we read the pdf we convert it into the vector embeddings 
#     from langchain_google_genai import GoogleGenerativeAIEmbeddings 
#     import google.generativeai as genai

#     from langchain_community.vectorstores import FAISS

#     from langchain_google_genai import ChatGoogleGenerativeAI
#     from langchain.chains.question_answering import load_qa_chain 
#     from langchain.prompts import PromptTemplate 
#     from dotenv import load_dotenv 



#     load_dotenv()
#     genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
#     def get_pdf_text(doc):
#         text=""
#         for pdf in doc:
#             pdf_reader=PdfReader(pdf) #it gets the multiple pages 
#             for page in pdf_reader.pages:
#                 text+=page.extract_text()  #here we are extracting the text from the pdf's pages 
#         return text 

#     #lets divide the text into the chunks 
#     def get_txt_chunks(text):
#         text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
#         chunks=text_splitter.split_text(text)
#         return chunks
#     #lets convert the text chunks into the vectors 
#     def get_vector_store(text_chunks):
#         embeddings=GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
#         vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
#         vector_store.save_local('faiss_index')
#     def get_conversational_chain():
#         prompt_template='''answer the question as detailed as possible from the provided context, make sure to 
#         provide all the details, if the answer is not in provided context just say, "answer is not available in 
#         the context",don't provide the wrong answer \n\n
#         context:\n {context}?\n
#         Question: \n{question}\n
#         Answer:
#         '''
#         model=ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)
#         prompt=PromptTemplate(template=prompt_template,input_variables=['context','question'])
#         chain=load_qa_chain(model,chain_type='stuff',prompt=prompt)
#         return chain 
#     def user_input(user_question):
#         embeddings=GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
#         new_db=FAISS.load_local('faiss_index',embeddings)
#         docs=new_db.similarity_search(user_question)
#         chain=get_conversational_chain()
#         response=chain({'input_documents':docs,'question':user_question}
#                     ,return_only_outputs=True)
#         print(response)
#         st.write('Replay: ',response['output_text'])


#     #lets create the streamlit application 
#     def main():
#         st.set_page_config('Chat with multiple pdfs')
#         st.header('chat with pdf using gemini')
#         user_question=st.text_input('Ask a Question from the PDF Files')
#         if user_question:
#             user_input(user_question)
#         with st.sidebar:
#             st.title('Menu: ')
#             pdf_docs=st.file_uploader('Upload your PDF Files and click on the enter')
#             if st.button('Submit & Process'):
#                 with st.spinner('Processing...'):
#                     raw_text=get_pdf_text(pdf_docs)
#                     text_chunks=get_txt_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success('Done')

# if __name__=="__main__":
#         main()
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

# def user_input(user_question):
#     """Handles user input and fetches answers."""
#     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
   
#     new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    

#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({'input_documents': docs, 'question': user_question}, return_only_outputs=True)
    
#     st.write('Reply:', response.get('output_text', 'Error generating response'))
def user_input(user_question):
    """Handles user input and fetches answers."""
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({'input_documents': docs, 'question': user_question}, return_only_outputs=True)
    output_text = response.get('output_text', 'Error generating response')
    return output_text

#def main():

#     st.set_page_config(
#     page_title="Chat with PDFs",
#     page_icon="📄",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

#     st.header("Chat with PDFs")
    
#     user_question = st.text_input("Ask a question from the uploaded PDFs")
#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
        
#         if st.button("Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_txt_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("Processing complete! You can now ask questions.")
# def main():
#     st.set_page_config(
#         page_title="Chat with PDFs",
#         page_icon="📄",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )

#     # Custom CSS for additional styling
#     st.markdown("""
#         <style>
#             .main {
#                 background-color: #FAFAFA;
#                 padding: 20px;
#                 border-radius: 10px;
#             }
#             .sidebar .sidebar-content {
#                 background-color: #F0F2F6;
#             }
#             .stButton>button {
#                 color: white;
#                 background-color: #4CAF50;
#             }
#             .stTextInput>div>div>input {
#                 border: 1px solid #4CAF50;
#             }
#         </style>
#         """, unsafe_allow_html=True)

#     # Main content area
#     with st.container():
#         st.title("Chat with PDFs")
#         st.write("Upload your PDF documents and ask questions to interact with their content.")

#         # File uploader and processing
#         pdf_docs = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_txt_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("Processing complete! You can now ask questions.")

#         # User question input
#         user_question = st.text_input("Ask a question about the uploaded PDFs:")
#         if user_question:
#             user_input(user_question)

#     # Sidebar for additional information
#     with st.sidebar:
#         st.header("Instructions")
#         st.write("""
#             1. Upload one or more PDF documents using the uploader above.
#             2. Click on 'Submit & Process' to analyze the documents.
#             3. Once processing is complete, enter your question in the text box.
#             4. The application will provide answers based on the content of the uploaded PDFs.
#         """)

# def main():
#     st.set_page_config(
#         page_title="Chat with PDFs",
#         page_icon="📄",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )

#     # Custom CSS for global text alignment
#     st.markdown("""
#         <style>
#             .justified-text {
#                 text-align: justify;
#                 text-justify: inter-word;
#             }
#         </style>
#         """, unsafe_allow_html=True)

#     st.header("Chat with PDFs")

#     user_question = st.text_input("Ask a question from the uploaded PDFs")
#     if user_question:
#         response = user_input(user_question)
#         st.markdown(f'<div class="justified-text">{response}</div>', unsafe_allow_html=True)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

#         if st.button("Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_txt_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("Processing complete! You can now ask questions.")

def main():
    st.set_page_config(
        page_title="PDF Genie",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.header("Chat with PDFs")

    # Text input for user question
    user_question = st.text_input("Ask a question from the uploaded PDFs")

    # Button to submit the question
    if st.button("Generate Response"):
        if user_question:
            response = user_input(user_question)
            st.write('Reply:', response)
        else:
            st.warning("Please enter a question before generating a response.")

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
            else:
                st.warning("Please upload PDF files before processing.")

if __name__ == "__main__":
    main()
