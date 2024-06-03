# %%
# Standard and Third-Party Libraries
import os
import re
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Machine Learning and Natural Language Processing Libraries
from sentence_transformers import CrossEncoder
from sklearn.metrics import ndcg_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.nn import functional as F

# PDF Processing
from pypdf import PdfReader

# Text Splitting Utilities
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# OpenAI and Utilities
import openai
from openai import OpenAI
from helper_utils import word_wrap

# Load environment variables
_ = load_dotenv('.env')


# %%
import pandas as pd


def process_txt(filename):
    model_name = 'cross-encoder'
    # reader = PdfReader("./data/tesla10K.pdf")
    reader = PdfReader(filename)
    pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]

    # Split text by sentences
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0)
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
    return character_split_texts
# %%
# Function to clean and format each entry in the list
def clean_text_list(text_list):
    cleaned_texts = []
    for text in text_list:
        # Replace tab characters with a single space
        text = text.replace('\t', ' ')
        text = text.replace('\n', ' ')
        # Split text into lines and remove any leading/trailing whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Combine lines back into a single string with newline characters
        cleaned_text = '\n'.join(lines)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts


def rank_doc(query=None, text_chunks=None, topN=5):
    # Initialize the CrossEncoder model with the specified model name
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    if query is None or text_chunks is None:
        print('missing query or text chunk')

    # Predict scores for each document in relation to the query
    scores = reranker.predict([[query, doc] for doc in text_chunks])

    # Get indices of the top N scores in descending order
    top_indices = np.argsort(scores)[::-1][:topN]

    # Retrieve the top-ranked text documents using list indexing
    top_pairs = [text_chunks[index] for index in top_indices]
    return top_pairs  # Returns a list of the top-ranked text strings

openai.api_key = os.environ['OPENAI_API_KEY']
openai_client = OpenAI()
# %%
# def rag(query=None, pdf_file=None, model="gpt-3.5-turbo"):

#     character_split_texts = process_txt(pdf_file)
#     cleaned_texts = clean_text_list(character_split_texts)
#     retrieved_documents = rank_doc(query, cleaned_texts, topN=5)

#     information = "\n\n".join(retrieved_documents)

#     if query is None or retrieved_documents is None:
#         print('missing query or retrieved documents')

#     messages = [
#         {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
#     ]
    
#     response = openai_client.chat.completions.create(
#         model=model,
#         messages=messages,
#     )
#     content = response.choices[0].message.content
#     return content

import pickle

def rag( pdf_file=None, query=None):
    model="gpt-3.5-turbo"
    # Construct the filename for the pickle cache
    cache_dir = './cache'
    base_filename = os.path.basename(pdf_file)
    cleaned_texts_file = os.path.join(cache_dir, f"{base_filename}_cleaned_texts.pickle")

    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load cleaned texts from cache if available
    if os.path.exists(cleaned_texts_file):
        with open(cleaned_texts_file, 'rb') as f:
            print("Loading cleaned texts from cache.")
            cleaned_texts = pickle.load(f)
    else:
        # Process and clean texts if no cache is available
        character_split_texts = process_txt(pdf_file)
        cleaned_texts = clean_text_list(character_split_texts)
        # Cache the cleaned texts
        with open(cleaned_texts_file, 'wb') as f:
            pickle.dump(cleaned_texts, f)

    # Continue with document ranking and retrieval
    retrieved_documents = rank_doc(query, cleaned_texts, topN=5)
    information = "\n\n".join(retrieved_documents)

    if query is None or retrieved_documents is None:
        print('missing query or retrieved documents')

    messages = [
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]

    # Call the OpenAI API
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content


# %%
if __name__ == '__main__':
    query = "what is revenue for 2023?"
    pdf_file = "./data/tesla10K.pdf"
    output = rag(query, pdf_file)
    print(word_wrap(output))

# %%
