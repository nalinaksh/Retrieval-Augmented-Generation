from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import streamlit as st
import openai
import subprocess
import time

#Add lines below to make the app work on streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

###START To create and download embeddings, execute steps below on a GPU 
# from langchain.document_loaders import WebBaseLoader
# url = "https://www.globalgreyebooks.com/online-ebooks/paramhansa-yogananda_autobiography-of-a-yogi_complete-text.html"
# loader = WebBaseLoader(url)
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
# all_splits = text_splitter.split_documents(data)

#persist_dir = "HuggingFaceEmbeddings"

# model_name = "BAAI/bge-large-en"
# model_kwargs = {'device': 'cuda'}
# encode_kwargs = {'normalize_embeddings': False}
# embedding = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs)

#vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_dir)
#vectorstore.persist()
#use either jar or tar cmd to create archive files for embeddings
#!jar cvf HuggingFaceEmbeddings.jar HuggingFaceEmbeddings
#!tar cvzf HuggingFaceEmbeddings.tar.gz HuggingFaceEmbeddings
#Download above jar file on your local machine
###END

###START To use the saved embeddings on a CPU
#Extract saved embeddings here
#!jar xvf HuggingFaceEmbeddings.jar
#!tar xvzf HuggingFaceEmbeddings.tar.gz
cmdline = ['/bin/tar','xvzf','HuggingFaceEmbeddings.tar.gz']
subprocess.call(cmdline)
time.sleep(30)

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)

persist_dir = "HuggingFaceEmbeddings"
vectorstore = None
vectorstore = Chroma(embedding_function=embedding, persist_directory=persist_dir)
###END

question = ""

llm = ChatOpenAI(openai_api_key=openai.api_key, model_name="gpt-3.5-turbo", temperature=0)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qna_chain = ConversationalRetrievalChain.from_llm(llm,
                                                  vectorstore.as_retriever(),
                                                  memory=memory,
                                                  condense_question_prompt=CONDENSE_QUESTION_PROMPT)

def ask(query):
  result = qa_chain({"question": query})
  return result["answer"]
