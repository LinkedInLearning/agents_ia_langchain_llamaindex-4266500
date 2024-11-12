
# pip install langchain-openai
# pip install langchain-community
# pip install torch
# pip install sentence_transformers
# pip install faiss-cpu
# pip install docx2txt

import os
import argparse
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS


os.environ["TOKENIZERS_PARALLELISM"] = "false" 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, default="./data/")
    parser.add_argument("--persist_dir", type=str, default="data_faiss")
    args = parser.parse_args()

    print(f"Using data dir {args.docs_dir}")
    print(f"Using index path {args.persist_dir}")
    
    embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    print(f"Embedding: {embedding.model_name}")
 
    if os.path.exists(args.persist_dir): 
        print(f"Loading FAISS index from {args.persist_dir}")
        vectorstore = FAISS.load_local(args.persist_dir, embedding,  allow_dangerous_deserialization=True)
        print("done.")
    else:
        print(f"Building FAISS index from documents in {args.docs_dir}")
         
        loader = DirectoryLoader(args.docs_dir,
            loader_cls=Docx2txtLoader,
            recursive=True,
            silent_errors=True,
            show_progress=True,
            glob="**/*.docx"  # which files get loaded
        )
        docs = loader.load()
       
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75
        )
        frags = text_splitter.split_documents(docs)

        print(f"Poplulating vector store with {len(docs)} docs in {len(frags)} fragments")
        vectorstore = FAISS.from_documents(frags, embedding)
        print(f"Persisting vector store to: {args.persist_dir}")
        vectorstore.save_local(args.persist_dir)
        print(f"Saved FAISS index to {args.persist_dir}")
     

if __name__ == "__main__":
    main()