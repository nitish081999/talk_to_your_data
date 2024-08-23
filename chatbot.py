import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI,ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import create_sql_agent
import requests
from bs4 import BeautifulSoup
import tempfile
import os

def process_document(doc_source, source_type):
    if source_type == "URL":
        response = requests.get(doc_source)
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'application/pdf' in content_type:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                loader = PyPDFLoader(tmp_file.name)
        elif 'text/html' in content_type:
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w+") as tmp_file:
                tmp_file.write(text_content)
                loader = TextLoader(tmp_file.name)
        else:  # Treat as plain text
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w+") as tmp_file:
                tmp_file.write(response.text)
                loader = TextLoader(tmp_file.name)
    elif source_type == "PDF":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(doc_source.read())
            loader = PyPDFLoader(tmp_file.name)
    else:  # Text file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w+") as tmp_file:
            content = doc_source.getvalue().decode("utf-8")
            tmp_file.write(content)
            loader = TextLoader(tmp_file.name)
    
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    
    return db

def get_conversation_chain(vector_store):
    llm = OpenAI()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
    )
    return conversation_chain

def get_sql_chain(db):
    llm=ChatOpenAI(model='gpt-3.5-turbo',temperature=0)
    agent_executor=create_sql_agent(llm,db=db,agent_type='openai-tools',verbose=True)
    return agent_executor

st.title("Document Q&A with Chat History")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

api_key = st.text_input("Enter your OpenAI API key:", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    source_type = st.radio("Select input type:", ("File Upload", "URL", "Database"))

    if source_type == "File Upload":
        file_type = st.radio("Select file type:", ("PDF", "Text"))
        uploaded_file = st.file_uploader(f"Choose a {file_type} file", type="pdf" if file_type == "PDF" else "txt")
        if uploaded_file is not None:
            vector_store = process_document(uploaded_file, file_type)
            conversation_chain = get_conversation_chain(vector_store)
    elif source_type == "URL":
        url = st.text_input("Enter the URL:")
        if url:
            vector_store = process_document(url, "URL")
            conversation_chain = get_conversation_chain(vector_store)
    elif source_type=='Database':  # Database
        uploaded_file = st.file_uploader("Choose a SQLite database file", type="db")
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                db_path = tmp_file.name
            
            db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
            
            sql_chain = get_sql_chain(db)


    user_question = st.text_input("Ask a question about the document or database:")
    
    if user_question:
        if source_type == "Database" and 'sql_chain' in locals():
            response = sql_chain.invoke(user_question)
            st.write("Answer:", response['output'])
        elif 'conversation_chain' in locals():
            response = conversation_chain({"question": user_question, "chat_history": st.session_state.chat_history})
            st.write("Answer:", response["answer"])
        
        st.session_state.chat_history.append((user_question, response['output']))
        
    st.subheader("Chat History")
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.write(f"Q{i+1}: {question}")
        st.write(f"A{i+1}: {answer}")
        st.write("---")
else:
    st.warning("Please enter your OpenAI API key to proceed.")