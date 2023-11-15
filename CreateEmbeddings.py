###START To create and download embeddings, execute steps below on a GPU 
# from langchain.document_loaders import WebBaseLoader
# url = "https://www.globalgreyebooks.com/online-ebooks/paramhansa-yogananda_autobiography-of-a-yogi_complete-text.html"
# loader = WebBaseLoader(url)
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
# all_splits = text_splitter.split_documents(data)

#persist_dir = "HuggingFaceEmbeddings"

# model_name = "BAAI/bge-large-en-v1.5"
# model_kwargs = {'device': 'cuda'}
# encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# embedding = HuggingFaceBgeEmbeddings(
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
