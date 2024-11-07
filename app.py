
# Import the required packages
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import bcrypt
import csv
import logging
from typing import Optional, List, Dict
import hashlib
import secrets


# Set OpenAI API key
api_key = os.environ['OA_API']           
os.environ['OPENAI_API_KEY'] = api_key

# Load LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Helper functions for user management
def hash_password(password: str) -> str:
  salt = secrets.token_hex(16)
  pw_hash = hashlib.sha256((password + salt).encode()).hexdigest()
  return f"{salt}:{pw_hash}"


def check_password(password: str, stored_hash: str) -> bool:
  try:
    salt, pw_hash = stored_hash.split(':')
    return hashlib.sha256((password + salt).encode()).hexdigest() == pw_hash
  except Exception:
    return False


def save_user(username: str, password: str, user_type: str) -> None:
  file_exists = os.path.isfile('users.csv')
  hashed_pw = hash_password(password)
    
  with open('users.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
      writer.writerow(['username', 'password', 'user_type'])
    writer.writerow([username, hashed_pw, user_type])


def authenticate_user(username: str, password: str)-> Optional[str]:
  if not os.path.exists('users.csv'):
    return None
        
  df = pd.read_csv('users.csv')
  if username not in df['username'].values:
    return None
        
  user = df[df['username'] == username].iloc[0]
  if check_password(password, user['password']):
    return user['user_type']
  return None


def load_allowed_docs(user_type: str)-> List:
  docs = []
  if user_type == 'admin':
    files = ["/content/drive/MyDrive/AI Datasets/finance_data.pdf", "/content/drive/MyDrive/AI Datasets/public_data.pdf"]
  else:
    files = ["/content/drive/MyDrive/AI Datasets/public_data.pdf"]
    
  for file in files:
    if not os.path.exists(file):
      raise FileNotFoundError(f"Required document {file} not found")
    loader = PyPDFLoader(file)
    docs.extend(loader.load())
    
  return docs


def initialize_rag(docs: List)-> ConversationalRetrievalChain:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="mmr", k=3),
        return_source_documents=True
    )
    
    return qa_chain

# Streamlit UI
def main():
    st.title("RAG System with User Access Control")

    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_type = None

    if not st.session_state.authenticated:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    user_type = authenticate_user(username, password)
                    if user_type:
                        st.session_state.authenticated = True
                        st.session_state.user_type = user_type
                        st.success(f"Logged in as {user_type}")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("signup"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                user_type = st.selectbox("User Type", ["admin", "researcher", "end-user"])
                signup_submitted = st.form_submit_button("Sign Up")
                
                if signup_submitted:
                    if not new_username or not new_password:
                        st.error("Username and password are required")
                    else:
                        save_user(new_username, new_password, user_type)
                        st.success("User created successfully!")

    else:
        st.write(f"Logged in as: {st.session_state.user_type}")
        
        if 'qa_chain' not in st.session_state:
            docs = load_allowed_docs(st.session_state.user_type)
            st.session_state.qa_chain = initialize_rag(docs)
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])

        if prompt := st.chat_input("Ask a question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)
            
            response = st.session_state.qa_chain({"question": prompt, "chat_history": []})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.chat_message("assistant").markdown(response["answer"])
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_type = None
            st.session_state.messages = []
            if 'qa_chain' in st.session_state:
                del st.session_state.qa_chain
            st.experimental_rerun()

if __name__ == "__main__":
    main()
