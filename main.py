import time
import math
import torch
import os
import PyPDF2
import numpy as np
from transformers import DPRContextEncoderTokenizerFast,DPRContextEncoder,DPRQuestionEncoder
from transformers import DPRQuestionEncoderTokenizerFast,RagRetriever,RagTokenizer,RagSequenceForGeneration
import faiss
import pandas as pd
from datasets import Dataset

torch.set_grad_enabled(False)

# Set the cache directory
cache_dir = "D:/Projects/QNA"

# Create the cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=100):  # Chunk Size set to 100 for easy divison, bert uses 512 maximum.
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

pdf_path = 'Frankenstein Book Report.pdf'
text = extract_text_from_pdf(pdf_path)
text_chunks = chunk_text(text)


ctx_tokenizer=DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base") 

#Tokenize the context (reference data)

num_passages=len(text_chunks)

outputs=ctx_tokenizer(
    text_chunks,
    truncation=True,
    padding='longest',
    return_tensors='pt',
)

input_ids=outputs["input_ids"]


ctx_encoder=DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

#Encode the Context

outputs=ctx_encoder(input_ids,return_dict=True)
embeddings=outputs["pooler_output"]


dim=768
m=128

index=faiss.IndexHNSWFlat(dim,m,faiss.METRIC_INNER_PRODUCT)
index.train(embeddings)
index.add(embeddings)

#Setting up Question Encoder and Tokenizer

q_encoder=DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
q_tokenizer=DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-multiset-base")
'''
input_ids=q_tokenizer.encode("Who is the author?",return_tensors="pt")
outputs=q_encoder(input_ids)

q_embed=outputs['pooler_output']

D,I=index.search(q_embed,k=3)
'''

#Creating custom dataset to feed into the RAG Model

df = pd.DataFrame({'text': text_chunks})
df.insert(0,'title','name')

dataset=Dataset.from_pandas(df)

embs=[]
for i in range(embeddings.shape[0]):
    embs.append(embeddings[i,:])

embs_np=[emb.numpy() for emb in embs]
dataset=dataset.add_column("embeddings",embs_np)
print(dataset)

index=faiss.IndexHNSWFlat(dim,m,faiss.METRIC_INNER_PRODUCT)
dataset.add_faiss_index(column="embeddings",index_name="embeddings",custom_index=index,faiss_verbose=True)

#Creating Retriever Model

retriever=RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    use_dummy_dataset=False,
    indexed_dataset=dataset,
    index_name="embeddings"
)

tokenizer=RagTokenizer.from_pretrained(
    "facebook/rag-sequence-nq"
)

model=RagSequenceForGeneration.from_pretrained(
    "facebook/rag-sequence-nq",
    retriever=retriever
)

def ask_question(question):
    qstarttime=time.time()
    input_ids=tokenizer.question_encoder(question,return_tensors="pt")["input_ids"]
    generated=model.generate(input_ids)
    generated_string=tokenizer.batch_decode(generated,skip_special_tokens=True)[0]
    print("----------------------------------")
    print("Q:",question)
    print("A:",generated_string)
    qendtime=time.time()
    print("Time taken in seconds:",(qendtime-qstarttime))

while(True):
    print("------------------------------------")
    print("Ask a question.")
    
    question=input()
    print("--------Thinking...------------")
    ask_question(question)
    print("-------------------")