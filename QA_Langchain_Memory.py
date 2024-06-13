from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from CreateHFEmbedding import *
import streamlit as st
import time
import langchain
import subprocess
import os

#Set to True for debugging
langchain.verbose = False

#Add lines below to make the app work on streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Set OpenAI API key from Streamlit secrets
# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

###START To create and download embeddings, execute steps below on a GPU 
#url = "https://www.globalgreyebooks.com/online-ebooks/paramhansa-yogananda_autobiography-of-a-yogi_complete-text.html"
#persist_dir = "HuggingFaceEmbeddings"
#create_embedding_vectorstore(url, persist_dir)
#use tar cmd to create archive files for embeddings
#!tar cvzf HuggingFaceEmbeddings.tar.gz HuggingFaceEmbeddings
#Download above jar file on your local machine
###END

###START To use the saved embeddings on a CPU
#Extract saved embeddings here
cmdline = ['/bin/tar','xvzf','HuggingFaceEmbeddings.tar.gz']
subprocess.call(cmdline)
time.sleep(2)

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

@st.cache_resource
def get_embedding():
    embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)
    return embedding

persist_dir = "HuggingFaceEmbeddings"

@st.cache_resource
def get_vectorstore():
    vectorstore = Chroma(embedding_function=get_embedding(), persist_directory=persist_dir)
    return vectorstore
###END

@st.cache_resource
def get_chat_openai():
    llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                     model_name="gpt-3.5-turbo", 
                     temperature=0,
                     max_tokens=200,
                     streaming=True, 
                     callbacks=[StreamingStdOutCallbackHandler()]
    )
    return llm

@st.cache_resource
def get_question_generator():
    question_generator = LLMChain(llm=get_chat_openai(), prompt=CONDENSE_QUESTION_PROMPT)
    return question_generator

prompt_template = """You are a helpful assistant. Use the context below to concisely answer the question at the end. \
The context is from the book - "Autobiography of a Yogi", written by Paramahansa Yogananda. \
If you do not find the answer in the given context, just say that you don't know the answer. \
Do not try to make up the answer. Do not use any external source such as internet to look up for the answer. 

Context:
{context}

Question: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate.from_template(prompt_template)

@st.cache_resource
def get_doc_chain():
    doc_chain = load_qa_chain(get_chat_openai(), chain_type="stuff", prompt=QA_PROMPT)
    return doc_chain
    
@st.cache_resource
def get_memory():
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
    return memory
    
vectorstore = get_vectorstore()

@st.cache_resource
def get_chain():
    chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        question_generator=get_question_generator(),
        combine_docs_chain=get_doc_chain(),
        memory=get_memory(),
        verbose=False # Set to True for debugging
    )
    return chain

chain = get_chain()

def ask(query):
  result = chain({"question": query})
  return result["answer"]
