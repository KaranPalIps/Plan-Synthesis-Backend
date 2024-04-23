import os
import json
from flask import Flask, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename

import pinecone
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone as pc
from pinecone import Pinecone
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from threading import Thread
from flask_cors import CORS
from langchain.embeddings.ollama import OllamaEmbeddings
import concurrent.futures
import chromadb



load_dotenv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = ['pdf']
CORS(app)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


Question = [
    'What is the last review date?',
    'What is the effective date?',
    'What are the HCPCS Code?',
    'What is policy number?',
    'What is policy name?',
    'What is the last review date?',
    'What is the email address?',
    'What is the fax number?',
    'What is the address?',
    'What is the phone number?',
    'What is the length of approval?'
]

@app.route('/healthChecker', methods=['GET'])
def healthChecker():
    return jsonify(message='Healthy')

def process_question(question_index, question, vectordb):
    try:
        mistral_result = mistralChat(question, vectordb)
        llama_result = llamaChat(question, vectordb)
        gemma_result = gemmaChat(question, vectordb)

        question_data = {
            f'Question {question_index}': question,
            'Mistral': mistral_result,
            'Llama': llama_result,
            'Gemma': gemma_result,
        }
        return question_data
    except Exception as e:
        print(f"Error processing question {question_index}: {e}")
        # Consider returning a default value or error message here

@app.route('/embeddings', methods=['POST'])
def create_embeddings():
    try:
        # Check if file exists and is allowed
        max_threads = 5  # Adjust as needed
        if 'files' not in request.files:
            return jsonify(message='No file part'), 400

        file = request.files['files']
        if file.filename == '':
            return jsonify(message='No selected file'), 400
        documents = []
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # load the file
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=[
                    "\n\n",
                    "\n",
                ]
            )
            texts = text_splitter.split_documents(documents)
            print('Vector Stores')
            persist_directory = 'vertorStore/db'
            try:
                vectorstore = Chroma.from_documents(
                    documents=texts,
                    persist_directory=persist_directory,
                    embedding=OllamaEmbeddings(model='nomic-embed-text'),
                )
            except Exception as e:
                print(e)
            max_threads = 5  # You can adjust this value as needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(process_question, i, question, vectorstore) for i, question in enumerate(Question)]
                result = []
                for future in concurrent.futures.as_completed(futures):
                    result.append(future.result()) 
            vectorstore.delete_collection()
            vectorstore.persist()

            return result, 200

        return jsonify(message='Allowed file types are ' + ', '.join(app.config['ALLOWED_EXTENSIONS']))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def mistralChat(prompt, vectordb):
    try:
        llm = ChatGroq(temperature=1, model="mixtral-8x7b-32768",api_key=os.getenv('GROQ_API_KEY'))
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
        )

        PROMPT_TEMPLATE = """You are a good assistant that answer questions. Your knowledge is strictly limited to the following piece of context. Use it to answer the question at the end.
        If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer question that are related to the context.
        Give a response in the same language as the question.
    
        Context: {context}
        """
        system_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_prompt

        ai_response = chain({"question": prompt, "chat_history": ''})
        return ai_response
    except Exception as e:
        print('mistralChat',e)


def llamaChat(prompt, vectordb):
    try:
        llm = ChatGroq(temperature=1, model="llama3-70b-8192",api_key=os.getenv('GROQ_API_KEY1'))
        DB_PATH = "vectorstores/db/"
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
        )

        PROMPT_TEMPLATE = """You are a good assistant that answer questions. Your knowledge is strictly limited to the following piece of context. Use it to answer the question at the end.
        If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer question that are related to the context.
        Give a response in the same language as the question.
        
        Context: {context}
        """
        system_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_prompt

        ai_response = chain({"question": prompt, "chat_history": ''})

        return ai_response
    except Exception as e:
        print('llamaChat',e)


def gemmaChat(prompt, vectordb):
    try:
        llm = ChatGroq(temperature=1, model="gemma-7b-it",api_key=os.getenv('GROQ_API_KEY2'))
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
        )

        PROMPT_TEMPLATE = """You are a good assistant that answer questions. Your knowledge is strictly limited to the following piece of context. Use it to answer the question at the end.
        If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer question that are related to the context.
        Give a response in the same language as the question.
        
        Context: {context}
        """
        system_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_prompt

        ai_response = chain({"question": prompt, "chat_history": ''})

        return ai_response
    except Exception as e:
        print('gemmaChat',e)



if __name__ == '__main__':
    app.run(debug=True)
