from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
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

from CreateHFEmbedding import *
import streamlit as st
import openai
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
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

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
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)

persist_dir = "HuggingFaceEmbeddings"
vectorstore = None
vectorstore = Chroma(embedding_function=embedding, persist_directory=persist_dir)
###END

llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], 
                 model_name="gpt-3.5-turbo", 
                 temperature=0,
                 max_tokens=200,
                 streaming=True, 
                 callbacks=[StreamingStdOutCallbackHandler()]
)

question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

prompt_template = """Use the following pieces of context to answer the question at the end. \
If you could not find the answer from the given context, just say that you don't know the answer. \
Don't consult any other external source to look up or make up the answer. \
Keep the answer as concise as possible. End all of your responses with "Jai Guru 🙏"

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

#Something like this can be used directly retrieve docs from the vectorstore for a given query. 
# Not required as of now 
def get_relevant_docs(query):
    m = memory.load_memory_variables({})
    chat_history = m['chat_history']
    standalone_question = question_generator({"question":query, "chat_history":chat_history})
    context = "\n\nContext: "
    docs = vectorstore.similarity_search(standalone_question)
    for doc in docs:
        context += "\n" + doc.page_content
    return context
    
def ask(query):
  result = chain({"question": query})
  return result["answer"]
            

def get_vecstore():
    return vectorstore
