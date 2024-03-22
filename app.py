import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import pinecone
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone as pc
from pinecone import Pinecone
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from backend import response, llamaresponse, gemmaresponse
from flask_cors import CORS

load_dotenv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = ['pdf']
CORS(app)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# @app.route('/embeddings', methods=['POST'])
# def create_embeddings():
#     # Check if file exists and is allowed
#     print('request', request.files)
#     if 'files' not in request.files:
#         return jsonify(message='No file part'), 400
#     file = request.files['files']
#     if file.filename == '':
#         return jsonify(message='No selected file'), 400
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         print('filename', filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         # load the file
#         loader = PyPDFLoader(filepath)

#         # split into chunks
#         data = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=2000, chunk_overlap=0)
#         texts = text_splitter.split_documents(data)

#         # set up the embeddings object
#         embeddings = OllamaEmbeddings(model='nomic-embed-text')

#         # initialize and upload embeddings to Pinecone
#         Pinecone(
#             api_key=os.getenv('PINECONE_API_KEY'),
#             environment=os.getenv('PINECONE_API_ENV')
#         )
#         index_name = "plan-synthesis"  # replace with your index name

#         # upload to our pinecone index
#         pc.from_texts([t.page_content for t in texts],
#                       embeddings, index_name=index_name)
#         data = {
#             "message": "File uploaded successfully",
#             "filename": filename
#         }
#         return json.dumps(data)
#     return jsonify(message='Allowed file types are ' + ', '.join(app.config['ALLOWED_EXTENSIONS'])), 400


@app.route('/chat', methods=['POST'])
def create_chat():
    payload = request.get_json()
    
    if 'message' not in payload:
        return jsonify(message='No message provided'), 400
    
    if 'pdf' not in payload:
        return jsonify(message='No pdf path provided'), 400
    filename = payload['pdf']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    message = payload['message']
    return gemmaresponse(message, pdf=filepath)


@app.route('/llamachat', methods=['POST'])
def llama_Chat():
    payload = request.get_json()
    
    if 'message' not in payload:
        return jsonify(message='No message provided'), 400
    
    if 'pdf' not in payload:
        return jsonify(message='No pdf path provided'), 400
    filename = payload['pdf']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    message = payload['message']
    return llamaresponse(message, pdf=filepath)

@app.route('/gemmachat', methods=['POST'])
def gemma_Chat():
    payload = request.get_json()
    
    if 'message' not in payload:
        return jsonify(message='No message provided'), 400
    
    if 'pdf' not in payload:
        return jsonify(message='No pdf path provided'), 400
    filename = payload['pdf']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    message = payload['message']
    return llamaresponse(message, pdf=filepath)


if __name__ == '__main__':
    app.run(debug=True)
