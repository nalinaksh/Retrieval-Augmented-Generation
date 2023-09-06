from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import openai

openai.api_key = "xxxx"

# Set env var OPENAI_API_KEY or load from a .env file
# import dotenv
# dotenv.load_env()

from langchain.document_loaders import WebBaseLoader
url = "https://www.globalgreyebooks.com/online-ebooks/paramhansa-yogananda_autobiography-of-a-yogi_complete-text.html"
loader = WebBaseLoader(url)
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
all_splits = text_splitter.split_documents(data)

#To persist embeddings 
#embedding=OpenAIEmbeddings(openai_api_key=openai.api_key)
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)

### To create and download embeddings for later use on a CPU, execute steps below on a GPU 
#persist_dir = "HuggingFaceEmbeddings"
#vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_dir)
#vectorstore.persist()
#!jar cvf HuggingFaceEmbeddings.jar HuggingFaceEmbeddings
#Download above jar file on your local machine
###END

vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)

### To load and use downloaded embeddings on a CPU, execute steps below
#!jar xvf HuggingFaceEmbeddings.jar

#model_name = "BAAI/bge-large-en"
#model_kwargs = {'device': 'cpu'} <--IMP!!! device has to be changed to CPU for embeddings to be used on a CPU
#encode_kwargs = {'normalize_embeddings': False}
#embedding = HuggingFaceEmbeddings(
    #model_name=model_name,
    #model_kwargs=model_kwargs,
    #encode_kwargs=encode_kwargs)

#vectorstore = None
#vectorstore = Chroma(embedding_function=embedding, persist_directory=persist_dir)
###END

question = ""

llm = ChatOpenAI(openai_api_key=openai.api_key, model_name="gpt-3.5-turbo", temperature=0)
#qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
#qa_chain({"query": question})

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

def ask(question):
  result = qa_chain({"query": question})
  return result["result"]
