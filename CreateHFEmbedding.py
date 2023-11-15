from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

def get_data_splits(url):
  loader = WebBaseLoader(url)
  data = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
  all_splits = text_splitter.split_documents(data)
  return all_splits

def get_embedding_fn():
  model_name = "BAAI/bge-large-en-v1.5"
  model_kwargs = {'device': 'cuda'}
  encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
  embedding = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
  return embedding

def create_embedding_vectorstore(url, persist_dir):
  all_splits = get_data_splits(url)
  embedding = get_embedding_fn()
  vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_dir)
  vectorstore.persist()
  
