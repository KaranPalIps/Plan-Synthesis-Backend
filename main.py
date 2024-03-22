import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import pinecone
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone as pc
from pinecone import Pinecone
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.indexes import VectorstoreIndexCreator
from dotenv import load_dotenv
from backend import response, llamaresponse, gemmaresponse
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from flask_cors import CORS
import chromadb

load_dotenv




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = ['pdf']
CORS(app)



pdf_folder_path = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'])
documents = []
for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
    add_start_index = True,)
texts = text_splitter.split_documents(documents)
 # set up the embeddings object
print('Emabedding the documents')
embeddings = OllamaEmbeddings(model='nomic-embed-text')

print('Initializing the pinecone')

        # initialize and upload embeddings to Pinecone
client = chromadb.Client()
if client.list_collections():
    consent_collection = client.create_collection("consent_collection")
else:
    print("Collection already exists")
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
    persist_directory="D:\\A.I projects\\Plan-Synthesis demo\\db"
)
vectordb.persist()


def load_chunk_persist_pdf() -> Chroma:
    pdf_folder_path = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'])
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("consent_collection")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
        persist_directory="D:\\A.I projects\\Plan-Synthesis demo\\db"
    )
    vectordb.persist()
    return vectordb


def create_agent_chain():
    llm = ChatGroq(temperature=1, model="mixtral-8x7b-32768",api_key=os.getenv('GROQ_API_KEY'))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
    retriever = vectordb.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= memory)
    return chain


def create_llama_chain():
    llm = ChatGroq(temperature=1, model="llama2-70b-4096",api_key=os.getenv('GROQ_API_KEY'))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
    retriever = vectordb.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= memory)
    return chain

def create_gemma_chain():
    llm = ChatGroq(temperature=1, model="gemma-7b-it",api_key=os.getenv('GROQ_API_KEY'))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
    retriever = vectordb.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= memory)
    return chain



@app.route('/chat', methods=['POST'])
def create_chat():
    payload = request.get_json()
    
    if 'message' not in payload:
        return jsonify(message='No message provided'), 400
    
    query = payload['message']
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run({'question': query})
    return answer


@app.route('/llamachat', methods=['POST'])
def llama_Chat():
    payload = request.get_json()
    
    if 'message' not in payload:
        return jsonify(message='No message provided'), 400
    
    query = payload['message']
    chain = create_llama_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run({'question': query})
    return answer

@app.route('/gemmachat', methods=['POST'])
def gemma_Chat():
    payload = request.get_json()
    
    if 'message' not in payload:
        return jsonify(message='No message provided'), 400
    
    query = payload['message']
    chain = create_gemma_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run({'question': query})
    return answer

if __name__ == '__main__':
    app.run(debug=True)
