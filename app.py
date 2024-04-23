import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import pinecone
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone as pc
from pinecone import Pinecone
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from flask_cors import CORS
from langchain.embeddings.ollama import OllamaEmbeddings
import chromadb



load_dotenv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = ['pdf']
CORS(app)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/healthChecker', methods=['GET'])
def healthChecker():
    return jsonify(message='Healthy')

def process_question(vectordb):
    try:
        mistral_result = mistralChat(vectordb)
        llama_result = llamaChat(vectordb)
        gemma_result = gemmaChat(vectordb)

        question_data = {
            'Mistral': mistral_result,
            'Llama': llama_result,
            'Gemma': gemma_result,
        }
        return question_data
    except Exception as e:
        print(f"Error processing: {e}")
        # Consider returning a default value or error message here

@app.route('/embeddings', methods=['POST'])
def create_embeddings():
    try:
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
            mistral_result = mistralChat(vectordb)
            llama_result = llamaChat(vectordb)
            gemma_result = gemmaChat(vectordb)

            question_data = {
                'Mistral': mistral_result,
                'Llama': llama_result,
                'Gemma': gemma_result,
            }

            question_data['Mistral'] = json.dumps(mistral_result)
            question_data['Llama'] = json.dumps(llama_result)
            question_data['Gemma'] = json.dumps(gemma_result)
            return question_data, 200

        return jsonify(message='Allowed file types are ' + ', '.join(app.config['ALLOWED_EXTENSIONS']))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def mistralChat(vectordb):
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

        prompt = """
        Questions:

        Question 1: What is the last review date?
        Question 2: What is the effective date?
        Question 3: What are the HCPCS Code?
        Question 4: What is policy number?
        Question 5: What is policy name?
        Question 6: What is the last review date?
        Question 7: What is the email address?
        Question 8: What is the fax number?
        Question 9: What is the address?
        Question 10: What is the phone number?
        Question 11: What is the effective date?
        Question 12: What is the length of approval?

        Give the response of the above question in json format
        """

        ai_response = chain({"question": prompt, "chat_history": ''})
        return ai_response
    except Exception as e:
        print('mistralChat',e)


def llamaChat(vectordb):
    try:
        llm = ChatGroq(temperature=1, model="llama2-70b-4096",api_key=os.getenv('GROQ_API_KEY'))
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

        prompt = """
        Questions:

        Question 1: What is the last review date?
        Question 2: What is the effective date?
        Question 3: What are the HCPCS Code?
        Question 4: What is policy number?
        Question 5: What is policy name?
        Question 6: What is the last review date?
        Question 7: What is the email address?
        Question 8: What is the fax number?
        Question 9: What is the address?
        Question 10: What is the phone number?
        Question 11: What is the effective date?
        Question 12: What is the length of approval?

        Give the response of the above question in json format
        """

        ai_response = chain({"question": prompt, "chat_history": ''})
        return ai_response
    except Exception as e:
        print('llamaChat',e)


def gemmaChat(vectordb):
    try:
        llm = ChatGroq(temperature=1, model="gemma-7b-it",api_key=os.getenv('GROQ_API_KEY'))
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

        prompt = """
        Answer the below questions from the context.
        Questions:

        Question 1: What is the last review date?
        Question 2: What is the effective date?
        Question 3: What are the HCPCS Code?
        Question 4: What is policy number?
        Question 5: What is policy name?
        Question 6: What is the last review date?
        Question 7: What is the email address?
        Question 8: What is the fax number?
        Question 9: What is the address?
        Question 10: What is the phone number?
        Question 11: What is the effective date?
        Question 12: What is the length of approval?

        Give the response of the above question in json format
        """

        ai_response = chain({"question": prompt, "chat_history": ''})
        return ai_response
    except Exception as e:
        print('gemmaChat',e)



if __name__ == '__main__':
    app.run(debug=True)
