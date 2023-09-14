from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import streamlit as st
import openai
import subprocess
import time
import langchain

#Set to True for debugging
langchain.verbose = False

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

llm = ChatOpenAI(openai_api_key=openai.api_key, 
                 model_name="gpt-3.5-turbo", 
                 temperature=0,
                 streaming=True, 
                 callbacks=[StreamingStdOutCallbackHandler()]
)

question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

prompt_template = """Use the following pieces of context to answer the question at the end. \
If you don't know the answer, just say that you don't know, don't try to make up an answer. \
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate.from_template(prompt_template)
doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_PROMPT)

memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    memory=memory,
    verbose=False # Set to True for debugging
)

def ask(query):
  result = chain({"question": query})
  return result["answer"]