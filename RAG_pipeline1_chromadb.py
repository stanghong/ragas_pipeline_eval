# %%
from pypdf import PdfReader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
import os
# %%
# Load environment variables
load_dotenv('.env')
# os.environ['OPENAI_API_KEY']
# %%

# Load PDF documents
def load_documents(file_path):
    reader = PdfReader(file_path)
    pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
    return pdf_texts

# Split documents into chunks
def chunk_documents(pdf_texts):
    # Split text by sentences
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0)
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

    # Tokenize the sentence chunks
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = [token_split_text for text in character_split_texts for token_split_text in token_splitter.split_text(text)]
    return token_split_texts


# Load PDF file
pdf_file = './data/tesla10K.pdf'

# %%
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import chromadb
# %%
def chromadb_retrieval_qa(pdf_file, question):
    # Load documents from the PDF
    data = load_documents(pdf_file)
    # Tokenize and split the text into chunks
    token_split_texts = chunk_documents(data)

    # Initialize embedding function
    embedding_function = SentenceTransformerEmbeddingFunction()
    cleaned_path = pdf_file.replace('.', '').replace('/', '')
    collection_name = cleaned_path

    # Initialize ChromaDB client
    chroma_client = chromadb.Client()

    try:
        # Try to retrieve the existing collection
        chroma_collection = chroma_client.get_collection(collection_name, embedding_function=embedding_function)
    except Exception as e:  # Replace with specific exception if known
        print(f"Collection not found or error retrieving: {str(e)}")
        # If the collection does not exist, create it
        chroma_collection = chroma_client.create_collection(collection_name, embedding_function=embedding_function)
        ids = [str(i) for i in range(len(token_split_texts))]
        chroma_collection.add(ids=ids, documents=token_split_texts)

    results = chroma_collection.query(query_texts=[question], n_results=5)
    retrieved_documents = results['documents'][0]
    openai_client = OpenAI()
    information = "\n\n".join(retrieved_documents)

    messages = [
        {"role": "user", "content": f"Question: {question}. \n Information: {information}"}
    ]
    model="gpt-3.5-turbo"
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content

    return content


# %%
if __name__ == '__main__':
    question = "summarize the text?"
    # question = "waht is tesla 2023 revenue"
    result = chromadb_retrieval_qa(pdf_file, question)
    print(result)


# %%
