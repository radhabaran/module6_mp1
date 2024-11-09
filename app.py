
# Import the required packages
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
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
import json


# Set OpenAI API key
api_key = os.environ['OA_API']           
os.environ['OPENAI_API_KEY'] = api_key

# Load LLMs
planner_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
classifier_llm = ChatOpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
qa_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Add directory creation for data and vector stores
def ensure_directories_exist():
    os.makedirs("data", exist_ok=True)
    os.makedirs("vector_stores", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

# Vector store paths
FINANCE_DB_PATH = "vector_stores/finance_db"
PUBLIC_DB_PATH = "vector_stores/public_db"

# Classification prompt
CLASSIFIER_TEMPLATE = """
Given a user query, determine which knowledge base(s) would be most relevant to answer it:
- Financial Data: Contains financial reports, metrics, and confidential business information
- Public Data: Contains general information, public reports, and non-confidential data

Query: {query}

Respond with a JSON object with the following structure:
{
    "financial_data": true/false,
    "public_data": true/false,
    "reasoning": "brief explanation"
}
"""

CLASSIFIER_PROMPT = PromptTemplate(
    input_variables=["query"],
    template=CLASSIFIER_TEMPLATE
)

# Planning prompt
PLANNER_TEMPLATE = """
Based on the classification results and user type, determine the final search strategy.

Classification Results: {classification_results}
User Type: {user_type}

Respond with a JSON object with the following structure:
{
    "search_financial": true/false,
    "search_public": true/false,
    "explanation": "brief explanation"
}

Remember:
- Only admin users can access financial data
- All users can access public data
"""

PLANNER_PROMPT = PromptTemplate(
    input_variables=["classification_results", "user_type"],
    template=PLANNER_TEMPLATE
)

class QueryPlanner:
    def __init__(self):
        self.classifier_chain = LLMChain(
            llm=classifier_llm,
            prompt=CLASSIFIER_PROMPT
        )
        self.planner_chain = LLMChain(
            llm=planner_llm,
            prompt=PLANNER_PROMPT
        )
    
    def classify_query(self, query: str) -> Dict:
        """Classify which knowledge base(s) might be relevant"""
        result = self.classifier_chain.run(query=query)
        return json.loads(result)
    
    def plan_search(self, classification_results: Dict, user_type: str) -> Dict:
        """Plan the final search strategy based on classification and user type"""
        result = self.planner_chain.run(
            classification_results=json.dumps(classification_results),
            user_type=user_type
        )
        return json.loads(result)

def create_vector_store(file_path: str, store_path: str):
    """Create or load a vector store using Chroma"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if os.path.exists(store_path):
        return Chroma(
            persist_directory=store_path,
            embedding_function=embeddings
        )
    
    # Create new vector store
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=store_path
    )
    return vectorstore



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
  file_exists = os.path.isfile('data/users.csv')
  hashed_pw = hash_password(password)
    
  with open('data/users.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
      writer.writerow(['username', 'password', 'user_type'])
    writer.writerow([username, hashed_pw, user_type])


def authenticate_user(username: str, password: str)-> Optional[str]:
  if not os.path.exists('data/users.csv'):
    return None
        
  df = pd.read_csv('data/users.csv')
  if username not in df['username'].values:
    return None
        
  user = df[df['username'] == username].iloc[0]
  if check_password(password, user['password']):
    return user['user_type']
  return None

class VectorStoreManager:
    def __init__(self, user_type: str):
        self.user_type = user_type
        self.finance_store = None
        self.public_store = None
        self.query_planner = QueryPlanner()
        
        # Initialize available stores based on user type
        self.initialize_stores()
    
    def initialize_stores(self):
        self.public_store = create_vector_store(
            "docs/public_data.pdf", 
            os.path.join("vector_stores", "public_db")
        )
        if self.user_type == 'admin':
            self.finance_store = create_vector_store(
                "docs/finance_data.pdf",
                os.path.join("vector_stores", "finance_db")
            )
    
    def get_relevant_documents(self, query: str) -> List:
        """Get relevant documents based on query classification and user type"""
        # Classify query
        classification = self.query_planner.classify_query(query)
        
        # Plan search strategy
        search_plan = self.query_planner.plan_search(classification, self.user_type)
        
        # Execute search
        documents = []
        if search_plan["search_financial"] and self.finance_store:
            finance_docs = self.finance_store.similarity_search(query, k=2)
            documents.extend(finance_docs)
        
        if search_plan["search_public"]:
            public_docs = self.public_store.similarity_search(query, k=2)
            documents.extend(public_docs)
        
        return documents


def initialize_rag(vector_store_manager: VectorStoreManager) -> ConversationalRetrievalChain:
    """Initialize RAG with intelligent query routing"""
    def smart_retriever(query):
        return vector_store_manager.get_relevant_documents(query)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=qa_llm,
        retriever=smart_retriever,
        return_source_documents=True,
        memory=None
    )
    
    return qa_chain


# Streamlit UI
def main():
    
    ensure_directories_exist()
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
                        st.rerun()
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
            pass
    else:
        st.write(f"Logged in as: {st.session_state.user_type}")
        
        if 'vector_store_manager' not in st.session_state:
            st.session_state.vector_store_manager = VectorStoreManager(st.session_state.user_type)

        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = initialize_rag(st.session_state.vector_store_manager)
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])

        if prompt := st.chat_input("Ask a question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)
            
            response = st.session_state.qa_chain({
                "question": prompt, 
                "chat_history": st.session_state.chat_history
            })
            st.session_state.chat_history.append((prompt, response["answer"]))
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.chat_message("assistant").markdown(response["answer"])
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_type = None
            st.session_state.messages = []
            st.session_state.chat_history = []
            if 'qa_chain' in st.session_state:
                del st.session_state.qa_chain
            if 'vector_store_manager' in st.session_state:
                del st.session_state.vector_store_manager
            st.rerun()

if __name__ == "__main__":
    main()
