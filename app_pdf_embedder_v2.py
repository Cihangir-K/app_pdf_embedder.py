import gradio as gr
from langchain_community.document_loaders import WebBaseLoader

from langchain_community.document_loaders import PyPDFLoader

from PyPDF2 import PdfReader
import os 

from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# what is the note from uploader?

def process_input(pdf_files, question):
    model_local =ChatOllama(model='mistral')

    # 1. Split data into chunks

    # urls = [
    #     "https://ollama.com/",
    #     "https://ollama.com/blog/windows-preview",
    #     "https://ollama.com/blog/openai-compatibility"
        
    # ]
    
    file_paths = [pdf_file.name for pdf_file in pdf_files]

    print("file_paths", file_paths)
    # pdfs_list = str(pdfs).split("\n")
    # print("pdfs_list", pdfs_list)

    # docs = [PyPDFLoader(pdfs_list).load() for pdfs in pdfs_list]
    docs = PdfReader(file_paths[0])
    # docs_list = [item for sublist in docs for item in sublist]
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    # docs_splits = text_splitter.split_documents(docs_list)

    # extract the text
    if docs is not None:
      pdf_reader = PdfReader(file_paths[0])
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    docs_splits = text_splitter.split_text(text)
    docs_splits2 = text_splitter.create_documents(docs_splits)
    print("docs_splits:", docs_splits)

    #2. Convert documents to Embeddings and store them


    vectorestore = Chroma.from_documents(
            documents=docs_splits2,
            collection_name="rag-chroma",
            embedding=embeddings.ollama.OllamaEmbeddings(model="nomic-embed-text"),persist_directory="./chroma_db",
    )
    retriever = vectorestore.as_retriever()

    

    #3. Before RAG

    # print ("Before RAG \n")
    # before_rag_template="what is {topic}"
    # before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
    # before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
    # print(before_rag_chain.invoke({"topic": "Ollama"}))

    #4. After RAG
    print('\n#######\nAfter RAG\n')
    after_rag_template = """"Answer the question based only on following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()

    )
    return after_rag_chain.invoke(question)

# Define Gradio interface 
iface = gr.Interface(fn=process_input,
                     inputs =[gr.File(file_count="multiple"),gr.Textbox(label="Question"),],
                     outputs="text",
                     title=" Document Query with Ollama",
                     description="Drop the file and a question to query documents.")
iface.launch()

    