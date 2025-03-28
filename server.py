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
import json

app = FastAPI()
load_dotenv()  # Add parentheses to execute the function
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
            
            # Ensure upload directory exists
            os.makedirs(config['UPLOAD_FOLDER'], exist_ok=True)
            
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

            # Set global settings
            Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
            Settings.chunk_size = 512

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
            
            print(question_data)
            return question_data

        return {"message": 'Allowed file types are ' + ', '.join(config['ALLOWED_EXTENSIONS'])}

    except Exception as e:
        print(f"Error in create_embeddings: {str(e)}")
        return {"error": str(e)}, 500


async def mistralChat(nodes, base_nodes, sentence_node_parser, base_node_parser):
    try:
        # Use API key from environment variables
        groq_api_key = os.getenv('MISTRAL_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        llm = Groq(temperature=1, model="mixtral-8x7b-32768", api_key=groq_api_key, response_format={"type": "json_object"})
        
        # Use Settings instead of ServiceContext
        Settings.llm = llm
        Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
        
        # Create the indices
        try:
            sentence_index = VectorStoreIndex(nodes)
            base_index = VectorStoreIndex(base_nodes)
        except Exception as e:
            print(f"Error creating indices: {e}")
            raise
            
        # Save indices
        sentence_index.storage_context.persist(persist_dir="./sentence_index")
        base_index.storage_context.persist(persist_dir="./base_index")
        
        # Retrieve from storage
        SC_retrieved_sentence = StorageContext.from_defaults(persist_dir="./sentence_index")
        SC_retrieved_base = StorageContext.from_defaults(persist_dir="./base_index")
        
        retrieved_sentence_index = load_index_from_storage(
            SC_retrieved_sentence, embed_model="local:BAAI/bge-small-en-v1.5")
        retrieved_base_index = load_index_from_storage(
            SC_retrieved_base, embed_model="local:BAAI/bge-small-en-v1.5")
        
        # Create query engine
        base_query_engine = retrieved_base_index.as_query_engine(
            similarity_top_k=5,
            verbose=True,
            llm=llm
        )
        
        print('Running Mistral inference')
        base_response = base_query_engine.query(question)
        
        # Parse the response into proper JSON
        try:
            # First try to parse it directly
            if hasattr(base_response, 'response'):
                json_response = json.loads(base_response.response)
            else:
                json_response = json.loads(str(base_response))
            return json_response
        except json.JSONDecodeError:
            # If the response is wrapped in ``` or other text, try to extract the JSON portion
            response_str = str(base_response)
            
            # Try to find JSON between triple backticks
            if "```" in response_str:
                parts = response_str.split("```")
                for i in range(len(parts)):
                    if i > 0 and (i % 2 == 1 or (parts[i-1].strip().endswith("json") or parts[i-1].strip() == "")):
                        try:
                            return json.loads(parts[i].strip())
                        except:
                            pass
            
            # If that fails, look for anything that might be JSON (between curly braces)
            try:
                start = response_str.find('{')
                end = response_str.rfind('}') + 1
                if start != -1 and end != 0:
                    return json.loads(response_str[start:end])
            except:
                pass
                
            # If all parsing attempts fail, return the string response with an error indicator
            return {"error": "Could not parse JSON response", "raw_response": str(base_response)}
    except Exception as e:
        print(f'Error in mistralChat: {e}')
        return f"Error in mistralChat: {str(e)}"


async def llamaChat(nodes, base_nodes, sentence_node_parser, base_node_parser):
    try:
        # Use API key from environment variables
        groq_api_key = os.getenv('LLAMA_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        llm = Groq(temperature=1, model="llama-3.3-70b-versatile", api_key=groq_api_key, response_format={"type": "json_object"})
        
        # Use Settings instead of ServiceContext
        Settings.llm = llm
        Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
        
        # Create the indices
        try:
            sentence_index = VectorStoreIndex(nodes)
            base_index = VectorStoreIndex(base_nodes)
        except Exception as e:
            print(f"Error creating indices: {e}")
            raise
            
        # Save indices
        sentence_index.storage_context.persist(persist_dir="./sentence_index")
        base_index.storage_context.persist(persist_dir="./base_index")
        
        # Retrieve from storage
        SC_retrieved_sentence = StorageContext.from_defaults(persist_dir="./sentence_index")
        SC_retrieved_base = StorageContext.from_defaults(persist_dir="./base_index")
        
        retrieved_sentence_index = load_index_from_storage(
            SC_retrieved_sentence, embed_model="local:BAAI/bge-small-en-v1.5")
        retrieved_base_index = load_index_from_storage(
            SC_retrieved_base, embed_model="local:BAAI/bge-small-en-v1.5")
        
        # Create query engine
        base_query_engine = retrieved_base_index.as_query_engine(
            similarity_top_k=5,
            verbose=True,
            llm=llm
        )
        
        print('Running Llama inference')
        base_response = base_query_engine.query(question)
        
        # Parse the response into proper JSON
        try:
            # First try to parse it directly
            if hasattr(base_response, 'response'):
                json_response = json.loads(base_response.response)
            else:
                json_response = json.loads(str(base_response))
            return json_response
        except json.JSONDecodeError:
            # If the response is wrapped in ``` or other text, try to extract the JSON portion
            response_str = str(base_response)
            
            # Try to find JSON between triple backticks
            if "```" in response_str:
                parts = response_str.split("```")
                for i in range(len(parts)):
                    if i > 0 and (i % 2 == 1 or (parts[i-1].strip().endswith("json") or parts[i-1].strip() == "")):
                        try:
                            return json.loads(parts[i].strip())
                        except:
                            pass
            
            # If that fails, look for anything that might be JSON (between curly braces)
            try:
                start = response_str.find('{')
                end = response_str.rfind('}') + 1
                if start != -1 and end != 0:
                    return json.loads(response_str[start:end])
            except:
                pass
                
            # If all parsing attempts fail, return the string response with an error indicator
            return {"error": "Could not parse JSON response", "raw_response": str(base_response)}
    except Exception as e:
        print(f'Error in llamaChat: {e}')
        return f"Error in llamaChat: {str(e)}"


async def gemmaChat(nodes, base_nodes, sentence_node_parser, base_node_parser):
    try:
        # Use API key from environment variables
        groq_api_key = os.getenv('GEMMA_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        llm = Groq(temperature=1, model="qwen-2.5-32b", api_key=groq_api_key, response_format={"type": "json_object"})
        
        # Use Settings instead of ServiceContext
        Settings.llm = llm
        Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
        
        # Create the indices
        try:
            sentence_index = VectorStoreIndex(nodes)
            base_index = VectorStoreIndex(base_nodes)
        except Exception as e:
            print(f"Error creating indices: {e}")
            raise
            
        # Save indices
        sentence_index.storage_context.persist(persist_dir="./sentence_index")
        base_index.storage_context.persist(persist_dir="./base_index")
        
        # Retrieve from storage
        SC_retrieved_sentence = StorageContext.from_defaults(persist_dir="./sentence_index")
        SC_retrieved_base = StorageContext.from_defaults(persist_dir="./base_index")
        
        retrieved_sentence_index = load_index_from_storage(
            SC_retrieved_sentence, embed_model="local:BAAI/bge-small-en-v1.5")
        retrieved_base_index = load_index_from_storage(
            SC_retrieved_base, embed_model="local:BAAI/bge-small-en-v1.5")
        
        # Create query engine
        base_query_engine = retrieved_base_index.as_query_engine(
            similarity_top_k=5,
            verbose=True,
            llm=llm
        )
        
        print('Running Gemma inference')
        base_response = base_query_engine.query(question)
        
        # Parse the response into proper JSON
        try:
            # First try to parse it directly
            if hasattr(base_response, 'response'):
                json_response = json.loads(base_response.response)
            else:
                json_response = json.loads(str(base_response))
            return json_response
        except json.JSONDecodeError:
            # If the response is wrapped in ``` or other text, try to extract the JSON portion
            response_str = str(base_response)
            
            # Try to find JSON between triple backticks
            if "```" in response_str:
                parts = response_str.split("```")
                for i in range(len(parts)):
                    if i > 0 and (i % 2 == 1 or (parts[i-1].strip().endswith("json") or parts[i-1].strip() == "")):
                        try:
                            return json.loads(parts[i].strip())
                        except:
                            pass
            
            # If that fails, look for anything that might be JSON (between curly braces)
            try:
                start = response_str.find('{')
                end = response_str.rfind('}') + 1
                if start != -1 and end != 0:
                    return json.loads(response_str[start:end])
            except:
                pass
                
            # If all parsing attempts fail, return the string response with an error indicator
            return {"error": "Could not parse JSON response", "raw_response": str(base_response)}
    except Exception as e:
        print(f'Error in gemmaChat: {e}')
        return f"Error in gemmaChat: {str(e)}"
