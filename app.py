import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging
import sys
from pprint import pprint
from llama_index.core import(
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    Document
)
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import Settings
from constant import question

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

load_dotenv

# Initialising the flask and app configs.
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = ['pdf']
CORS(app)

# The function that to only allow pdfs to be uploaded
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/healthChecker', methods=['GET'])
def healthChecker():
    return jsonify(message='Healthy')

def process_question(nodes, base_nodes, sentence_node_parser, base_node_parser):
    try:
        mistral_result = mistralChat(nodes, base_nodes, sentence_node_parser, base_node_parser)
        llama_result = llamaChat(nodes, base_nodes. sentence_node_parser, base_node_parser)
        gemma_result = gemmaChat(nodes, base_nodes, sentence_node_parser, base_node_parser)

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
            
            # Initializing the document loader with the chunks of the pdf
            documents = SimpleDirectoryReader(input_files=[filepath]).load_data()

            # create the sentence window parser w/ default settings
            
            sentence_node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=3,
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            )
        
            base_node_parser = SentenceSplitter()

            Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
            Settings.chunk_size = 512
            
            nodes = sentence_node_parser.get_nodes_from_documents(documents)
            base_nodes = base_node_parser.get_nodes_from_documents(documents)


            mistral_result = mistralChat(nodes, base_nodes, sentence_node_parser, base_node_parser)
            llama_result = llamaChat(nodes, base_nodes, sentence_node_parser, base_node_parser)
            gemma_result = gemmaChat(nodes, base_nodes, sentence_node_parser, base_node_parser)

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


def mistralChat(nodes, base_nodes, sentence_node_parser, base_node_parser):
    try:
        llm = Groq(temperature=1, model="mixtral-8x7b-32768",api_key=os.getenv('GROQ_API_KEY'))

        ctx_sentence = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=sentence_node_parser)
        ctx_base = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=base_node_parser)

        sentence_index = VectorStoreIndex(nodes, service_context=ctx_sentence)
        base_index = VectorStoreIndex(base_nodes, service_context=ctx_base)

        sentence_index.storage_context.persist(persist_dir="./sentence_index")
        base_index.storage_context.persist(persist_dir="./base_index")

        ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")

        # Retrieve from Storage
        SC_retrieved_sentence = StorageContext.from_defaults(persist_dir="./sentence_index")
        SC_retrieved_base = StorageContext.from_defaults(persist_dir="./base_index")

        retrieved_sentence_index = load_index_from_storage(SC_retrieved_sentence, embed_model="local:BAAI/bge-small-en-v1.5")
        retrieved_base_index = load_index_from_storage(SC_retrieved_base, embed_model="local:BAAI/bge-small-en-v1.5")

        # Create query engine
        sentence_query_engine = retrieved_sentence_index.as_query_engine(
            llm=llm,
            similarity_top_k=5,
            verbose=True,
            node_postprocessors = [
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )

        base_query_engine = retrieved_base_index.as_query_engine(
            similarity_top_k=5,
            verbose=True,
            llm=llm
        )

        #Inference
        base_response = sentence_query_engine(
            question
        )
        return base_response
    except Exception as e:
        print('mistralChat',e)


def llamaChat(nodes, base_nodes, sentence_node_parser, base_node_parser):
    try:
        llm = Groq(temperature=1, model="llama2-70b-4096",api_key=os.getenv('GROQ_API_KEY'))
        ctx_sentence = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=sentence_node_parser)
        ctx_base = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=base_node_parser)

        sentence_index = VectorStoreIndex(nodes, service_context=ctx_sentence)
        base_index = VectorStoreIndex(base_nodes, service_context=ctx_base)

        sentence_index.storage_context.persist(persist_dir="./sentence_index")
        base_index.storage_context.persist(persist_dir="./base_index")

        ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")

        # Retrieve from Storage
        SC_retrieved_sentence = StorageContext.from_defaults(persist_dir="./sentence_index")
        SC_retrieved_base = StorageContext.from_defaults(persist_dir="./base_index")

        retrieved_sentence_index = load_index_from_storage(SC_retrieved_sentence, embed_model="local:BAAI/bge-small-en-v1.5")
        retrieved_base_index = load_index_from_storage(SC_retrieved_base, embed_model="local:BAAI/bge-small-en-v1.5")

        # Create query engine
        sentence_query_engine = retrieved_sentence_index.as_query_engine(
            llm=llm,
            similarity_top_k=5,
            verbose=True,
            node_postprocessors = [
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )

        base_query_engine = retrieved_base_index.as_query_engine(
            similarity_top_k=5,
            verbose=True,
            llm=llm
        )

        #Inference
        base_response = sentence_query_engine(
            question
        )
        return base_response
    except Exception as e:
        print('llamaChat',e)


def gemmaChat(nodes, base_nodes, sentence_node_parser, base_node_parser):
    try:
        llm = Groq(temperature=1, model="gemma-7b-it",api_key=os.getenv('GROQ_API_KEY'))
        ctx_sentence = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=sentence_node_parser)
        ctx_base = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=base_node_parser)

        sentence_index = VectorStoreIndex(nodes, service_context=ctx_sentence)
        base_index = VectorStoreIndex(base_nodes, service_context=ctx_base)

        sentence_index.storage_context.persist(persist_dir="./sentence_index")
        base_index.storage_context.persist(persist_dir="./base_index")

        ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")

        # Retrieve from Storage
        SC_retrieved_sentence = StorageContext.from_defaults(persist_dir="./sentence_index")
        SC_retrieved_base = StorageContext.from_defaults(persist_dir="./base_index")

        retrieved_sentence_index = load_index_from_storage(SC_retrieved_sentence, embed_model="local:BAAI/bge-small-en-v1.5")
        retrieved_base_index = load_index_from_storage(SC_retrieved_base, embed_model="local:BAAI/bge-small-en-v1.5")

        # Create query engine
        sentence_query_engine = retrieved_sentence_index.as_query_engine(
            llm=llm,
            similarity_top_k=5,
            verbose=True,
            node_postprocessors = [
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )

        base_query_engine = retrieved_base_index.as_query_engine(
            similarity_top_k=5,
            verbose=True,
            llm=llm
        )

        #Inference
        base_response = sentence_query_engine(
            question
        )
        return base_response
    except Exception as e:
        print('gemmaChat',e)



if __name__ == '__main__':
    app.run(debug=True)
