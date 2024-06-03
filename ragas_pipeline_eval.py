# %%
# Libraries for environment and API access
import os
from dotenv import load_dotenv, find_dotenv
import openai
from openai import OpenAI

# Data handling and processing libraries
import pandas as pd
from datasets import Dataset

# PDF processing and text manipulation
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# Machine Learning and NLP libraries
from sentence_transformers import CrossEncoder
import numpy as np
from tqdm import tqdm

# Utilities and custom functions
from helper_utils import word_wrap
from RAG_pipeline1_chromadb import chromadb_retrieval_qa
from RAG_pipeline2_crossencoder import rag, rank_doc

# Metric evaluation
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy, faithfulness, context_recall, context_precision,
    context_relevancy, answer_correctness, answer_similarity
)

# Load environment variables
_ = load_dotenv('.env')
openai.api_key = os.environ['OPENAI_API_KEY']


# %%

# # Load Tesla 2023 10K report
pdf_file='./data/tesla10K.pdf'


def evaluate_ragas_dataset(ragas_dataset):
    result = evaluate(
        ragas_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
            context_relevancy,
            answer_correctness,
            answer_similarity
        ],
    )
    return result

# %%
def create_ragas_dataset(rag_pipeline, pdf_file, eval_dataset):
    rag_dataset = []
    for row in tqdm(eval_dataset):
        try:
            # Assuming rag_pipeline is a callable function that accepts a question
            # it needs to point to the text/pdf to run RAG on each row of questin
            answer = rag_pipeline(pdf_file, row["question"]) # Update based on your actual pipeline usage

            content = "No content available"
            contexts = ["No detailed context available"]

            # Check the type and contents of the answer
            if isinstance(answer, dict):
                content = answer.get('result', content)  # Safely get result from answer
                # Ensure 'context' is in answer and is a list before extracting
                if 'context' in answer and isinstance(answer['context'], list):
                    contexts = [context.page_content for context in answer['context']]

            elif isinstance(answer, str):
                # If answer is a string, directly use it as content
                content = answer

            # Append the collected data to rag_dataset
            rag_dataset.append({
                "question": row["question"],
                "answer": content,
                "contexts": [row['context']],
                "ground_truths": [row["ground_truth"]]
            })
        except Exception as e:
            print(f"Error processing row {row}: {e}")
            continue  # Optionally skip to next row or handle error differently

    # Convert the list of dictionaries to a DataFrame and then to an Arrow Dataset
    rag_df = pd.DataFrame(rag_dataset)
    rag_eval_dataset = Dataset.from_pandas(rag_df)
    return rag_eval_dataset

# %%
eval_dataset = Dataset.from_csv("./data/groundtruth_eval_dataset.csv")
# %%

ragas_dataset_pline1 = create_ragas_dataset(chromadb_retrieval_qa, pdf_file, eval_dataset )


# %%
evaluation_results_pline1 = evaluate_ragas_dataset(ragas_dataset_pline1)


# %%
df_pl1 = ragas_dataset_pline1.to_pandas()
df_pl1.to_excel('qc_metrics_pline1.xlsx')


# %%
# Create the RAGAS dataset
ragas_dataset_pline2 = create_ragas_dataset(rag, pdf_file, eval_dataset)

# %%
evaluation_results_pline2 = evaluate_ragas_dataset(ragas_dataset_pline2)
# %%
df_pl2 = ragas_dataset_pline2.to_pandas()
df_pl2.to_excel('qc_metrics_pline2.xlsx')
# evaluation_results_pipeline2 = evaluate_ragas_dataset(ragas_dataset)
# %%

result = pd.concat([df_pl1, df_pl2], axis=1)
result.to_excel('merged.xlsx')

# %%
# Create a DataFrame
df = pd.DataFrame({'ChromaDB ': evaluation_results_pline1, \
                   'ReRanker': evaluation_results_pline2})
df_transposed = df.transpose()


df_transposed
# %%
