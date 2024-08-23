import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI,ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
import requests
import pandas as pd
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

def excel_file_loader(file):
    df=pd.read_excel(file)
    agent=create_pandas_dataframe_agent(OpenAI(temperature=0),df)
    return agent
    

st.set_page_config(page_title="Interactive Document Q&A", layout="wide")

st.title("Talk to your data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Sidebar for API key and source selection
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your OpenAI API key:", type="password", value=st.session_state.api_key)
    if api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    source_type = st.radio("Select input type:", ("File Upload", "URL", "Database", "Excel file"))

# Main content area
if st.session_state.api_key:
    col1, col2 = st.columns([2, 1])

    with col1:
        if source_type == "File Upload":
            file_type = st.radio("Select file type:", ("PDF", "Text"))
            uploaded_file = st.file_uploader(f"Choose a {file_type} file", type="pdf" if file_type == "PDF" else "txt")
            if uploaded_file:
                with st.spinner("Processing document..."):
                    vector_store = process_document(uploaded_file, file_type)
                    conversation_chain = get_conversation_chain(vector_store)
                st.success("Document processed successfully!")

        elif source_type == "URL":
            url = st.text_input("Enter the URL:")
            if url:
                with st.spinner("Processing URL..."):
                    vector_store = process_document(url, "URL")
                    conversation_chain = get_conversation_chain(vector_store)
                st.success("URL processed successfully!")

        elif source_type == 'Database':
            uploaded_file = st.file_uploader("Choose a SQLite database file", type="db")
            if uploaded_file:
                with st.spinner("Processing database..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        db_path = tmp_file.name
                    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
                    sql_chain = get_sql_chain(db)
                st.success("Database processed successfully!")

        else:  # Excel file
            uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
            if uploaded_file:
                with st.spinner("Processing Excel file..."):
                    excel_agent = excel_file_loader(uploaded_file)
                st.success("Excel file processed successfully!")

        user_question = st.text_input("Ask a question about the document, database, or Excel file:")
        
        if user_question:
            with st.spinner("Generating answer..."):
                if source_type == "Database" and 'sql_chain' in locals():
                    response = sql_chain.invoke(user_question)
                    answer = response['output']
                elif 'conversation_chain' in locals():
                    response = conversation_chain({"question": user_question, "chat_history": st.session_state.chat_history})
                    answer = response["answer"]
                elif 'excel_agent' in locals():
                    answer = excel_agent.run(user_question)
                else:
                    answer = "Please process a document, database, or Excel file before asking questions."

            st.write("Answer:", answer)
            st.session_state.chat_history.append((user_question, answer))

    with col2:
        st.subheader("Chat History")
        for i, (question, answer) in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {question[:50]}..."):
                st.write(f"**Question:** {question}")
                st.write(f"**Answer:** {answer}")

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
