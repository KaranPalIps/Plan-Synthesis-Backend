import os
import bs4
import getpass
from langchain import hub
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders.pdf import PyPDFLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def response(user_query, pdf):

    # Load environment and get your openAI api key
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")


    # Select a webpage to load the context information from
    loader = PyPDFLoader(pdf)
    data = loader.load()


    # Restructure to process the info in chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="nomic-embed-text"))


    # Retrieve info from chosen source
    retriever = vectorstore.as_retriever(search_type="similarity")
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(temperature=1, model="mixtral-8x7b-32768",api_key=os.getenv('GROQ_API_KEY'))

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)



    template = """Use the following pieces of context to answer the question at the end.
    Say that you don't know when asked a question you don't know, donot make up an answer. Be precise and concise in your answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    # Add the context to your user query
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(user_query) 


def llamaresponse(user_query, pdf):

    # Load environment and get your openAI api key
    load_dotenv()


    # Select a webpage to load the context information from
    loader = PyPDFLoader(pdf)
    data = loader.load()


    # Restructure to process the info in chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="nomic-embed-text"))


    # Retrieve info from chosen source
    retriever = vectorstore.as_retriever(search_type="similarity")
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(temperature=1, model="llama2-70b-4096",api_key=os.getenv('GROQ_API_KEY'))

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)



    template = """Use the following pieces of context to answer the question at the end.
    Say that you don't know when asked a question you don't know, donot make up an answer. Be precise and concise in your answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    # Add the context to your user query
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(user_query) 


def gemmaresponse(user_query, pdf):

    # Load environment and get your openAI api key
    load_dotenv()


    # Select a webpage to load the context information from
    loader = PyPDFLoader(pdf)
    data = loader.load()


    # Restructure to process the info in chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="nomic-embed-text"))


    # Retrieve info from chosen source
    retriever = vectorstore.as_retriever(search_type="similarity")
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(temperature=1, model="gemma-7b-it",api_key=os.getenv('GROQ_API_KEY'))

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)



    template = """Use the following pieces of context to answer the question at the end.
    Say that you don't know when asked a question you don't know, donot make up an answer. Be precise and concise in your answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    # Add the context to your user query
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(user_query)