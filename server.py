from fastapi import UploadFile, FastAPI
import os
from dotenv import load_dotenv
import logging
import sys
from pprint import pprint
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    Document
)
from werkzeug.utils import secure_filename
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import Settings
from fastapi.middleware.cors import CORSMiddleware
from constant import question

app = FastAPI()
load_dotenv
config = {}
config['UPLOAD_FOLDER'] = 'uploads'
config['ALLOWED_EXTENSIONS'] = ['pdf']

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The function that to only allow pdfs to be uploaded
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config['ALLOWED_EXTENSIONS']


@app.get("/")
async def index():
    return {"message": "Hello World"}


@app.post("/embeddings")
async def create_embeddings(file: UploadFile):
    try:
        documents = []
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(file)
            filepath = os.path.join(config['UPLOAD_FOLDER'], filename)
            with open(filepath, "wb") as f:
                content = await file.read()  # Read the file content
                f.write(content)

            # Initializing the document loader with the chunks of the pdf
            documents = SimpleDirectoryReader(
                input_files=[filepath]).load_data()

            # create the sentence window parser w/ default settings
            sentence_node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=3,
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            )

            base_node_parser = SentenceSplitter()

            try:
                Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
                Settings.chunk_size = 512
                # Settings._llm=Groq(api_key=os.getenv('GROQ_API_KEY'))
            except Exception as e:
                print(e)

            nodes = sentence_node_parser.get_nodes_from_documents(documents)
            base_nodes = base_node_parser.get_nodes_from_documents(documents)

            mistral_result = await mistralChat(
                nodes, base_nodes, sentence_node_parser, base_node_parser)
            llama_result = await llamaChat(nodes, base_nodes, sentence_node_parser, base_node_parser)
            gemma_result = await gemmaChat(nodes, base_nodes, sentence_node_parser, base_node_parser)

            question_data = {
                'Mistral': mistral_result,
                'Llama': llama_result,
                'Gemma': gemma_result,
            }

            question_data['Mistral'] = mistral_result
            question_data['Llama'] = llama_result
            question_data['Gemma'] = gemma_result
            return question_data

        return {"message": 'Allowed file types are ' + ', '.join(config['ALLOWED_EXTENSIONS'])}

    except Exception as e:
        return {"error": str(e)}, 500


async def mistralChat(nodes, base_nodes, sentence_node_parser, base_node_parser):
    try:
        llm = Groq(temperature=1, model="mixtral-8x7b-32768",
                   api_key=os.getenv('GROQ_API_KEY'))
        ctx_sentence = ServiceContext.from_defaults(
            llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=sentence_node_parser)
        ctx_base = ServiceContext.from_defaults(
            llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=base_node_parser)
        try:
            sentence_index = VectorStoreIndex(
                nodes, service_context=ctx_sentence)
            base_index = VectorStoreIndex(base_nodes, service_context=ctx_base)
        except Exception as e:
            print(e)
        sentence_index.storage_context.persist(persist_dir="./sentence_index")
        base_index.storage_context.persist(persist_dir="./base_index")
        ServiceContext.from_defaults(
            chunk_size=1024, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
        # Retrieve from Storage
        SC_retrieved_sentence = StorageContext.from_defaults(
            persist_dir="./sentence_index")
        SC_retrieved_base = StorageContext.from_defaults(
            persist_dir="./base_index")
        retrieved_sentence_index = load_index_from_storage(
            SC_retrieved_sentence, embed_model="local:BAAI/bge-small-en-v1.5")
        retrieved_base_index = load_index_from_storage(
            SC_retrieved_base, embed_model="local:BAAI/bge-small-en-v1.5")
        # Create query engine
        sentence_query_engine = retrieved_sentence_index.as_query_engine(
            llm=Groq(model="mixtral-8x7b-32768",
                     api_key=os.getenv('GROQ_API_KEY')),
            similarity_top_k=5,
            verbose=True,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )

        base_query_engine = retrieved_base_index.as_query_engine(
            similarity_top_k=5,
            verbose=True,
            llm=Groq(model="mixtral-8x7b-32768",
                     api_key=os.getenv('GROQ_API_KEY'))
        )
        print('Inference')
        # Inference

        base_response = base_query_engine.query(question)

        return base_response
    except Exception as e:
        print('mistralChat', e)


async def llamaChat(nodes, base_nodes, sentence_node_parser, base_node_parser):
    try:
        llm = Groq(temperature=1, model="llama3-70b-8192",
                   api_key=os.getenv('GROQ_API_KEY'))
        ctx_sentence = ServiceContext.from_defaults(
            llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=sentence_node_parser)
        ctx_base = ServiceContext.from_defaults(
            llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=base_node_parser)
        try:
            sentence_index = VectorStoreIndex(
                nodes, service_context=ctx_sentence)
            base_index = VectorStoreIndex(base_nodes, service_context=ctx_base)
        except Exception as e:
            print(e)
        sentence_index.storage_context.persist(persist_dir="./sentence_index")
        base_index.storage_context.persist(persist_dir="./base_index")
        ServiceContext.from_defaults(
            chunk_size=1024, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
        # Retrieve from Storage
        SC_retrieved_sentence = StorageContext.from_defaults(
            persist_dir="./sentence_index")
        SC_retrieved_base = StorageContext.from_defaults(
            persist_dir="./base_index")
        retrieved_sentence_index = load_index_from_storage(
            SC_retrieved_sentence, embed_model="local:BAAI/bge-small-en-v1.5")
        retrieved_base_index = load_index_from_storage(
            SC_retrieved_base, embed_model="local:BAAI/bge-small-en-v1.5")
        # Create query engine
        sentence_query_engine = retrieved_sentence_index.as_query_engine(
            llm=Groq(model="llama3-70b-8192",
                     api_key=os.getenv('GROQ_API_KEY')),
            similarity_top_k=5,
            verbose=True,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )

        base_query_engine = retrieved_base_index.as_query_engine(
            similarity_top_k=5,
            verbose=True,
            llm=Groq(model="llama3-70b-8192",
                     api_key=os.getenv('GROQ_API_KEY'))
        )
        print('Inference')
        # Inference

        base_response = base_query_engine.query(question)

        return base_response
    except Exception as e:
        print('llamaChat', e)


async def gemmaChat(nodes, base_nodes, sentence_node_parser, base_node_parser):
    try:
        llm = Groq(temperature=1, model="gemma-7b-it",
                   api_key=os.getenv('GROQ_API_KEY'))
        ctx_sentence = ServiceContext.from_defaults(
            llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=sentence_node_parser)
        ctx_base = ServiceContext.from_defaults(
            llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", node_parser=base_node_parser)
        try:
            sentence_index = VectorStoreIndex(
                nodes, service_context=ctx_sentence)
            base_index = VectorStoreIndex(base_nodes, service_context=ctx_base)
        except Exception as e:
            print(e)
        sentence_index.storage_context.persist(persist_dir="./sentence_index")
        base_index.storage_context.persist(persist_dir="./base_index")
        ServiceContext.from_defaults(
            chunk_size=1024, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
        # Retrieve from Storage
        SC_retrieved_sentence = StorageContext.from_defaults(
            persist_dir="./sentence_index")
        SC_retrieved_base = StorageContext.from_defaults(
            persist_dir="./base_index")
        retrieved_sentence_index = load_index_from_storage(
            SC_retrieved_sentence, embed_model="local:BAAI/bge-small-en-v1.5")
        retrieved_base_index = load_index_from_storage(
            SC_retrieved_base, embed_model="local:BAAI/bge-small-en-v1.5")
        sentence_query_engine = retrieved_sentence_index.as_query_engine(
            llm=Groq(model="gemma-7b-it",
                     api_key=os.getenv('GROQ_API_KEY')),
            similarity_top_k=5,
            verbose=True,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )
        base_query_engine = retrieved_base_index.as_query_engine(
            similarity_top_k=5,
            verbose=True,
            llm=Groq(model="gemma-7b-it",
                     api_key=os.getenv('GROQ_API_KEY'))
        )
        print('Inference')
        # Inference

        base_response = base_query_engine.query(question)

        return base_response
    except Exception as e:
        print('gemmaChat', e)
