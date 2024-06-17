import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5Model
import faiss
import numpy as np
import torch
import json

def load_dotenv():
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    decoder_input_ids = tokenizer(" ", return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def load_faiss_index(index_file, dimension=512):
    if os.path.exists(index_file) and os.path.getsize(index_file) > 0:
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexFlatL2(dimension)  # Initialize a new index with the given dimension
    return index

def save_faiss_index(index, index_file):
    faiss.write_index(index, index_file)

def store_in_faiss(text, embedding, index, texts_file):
    embedding = np.array(embedding).astype('float32')
    index.add(np.expand_dims(embedding, axis=0))

    if os.path.exists(texts_file):
        with open(texts_file, 'r') as f:
            texts = json.load(f)
    else:
        texts = []

    texts.append(text)
    with open(texts_file, 'w') as f:
        json.dump(texts, f)

def search_similar_texts_faiss(query_text, model, tokenizer, index, texts, k=10):
    query_embedding = np.array(get_embedding(query_text, model, tokenizer)).astype('float32')
    query_embedding = np.expand_dims(query_embedding, axis=0)
    
    distances, indices = index.search(query_embedding, k)
    results = []
    for i in range(len(indices[0])):
        results.append((texts[indices[0][i]], distances[0][i]))
    
    # Sort results by distance (ascending), so that higher similarity appears first
    results.sort(key=lambda x: x[1])
    return results

def handle_input(user_input, index, texts_file, index_file):
    st.write("Processing your input...")
    # Load the model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5Model.from_pretrained('t5-small')

    # Get the embedding
    embedding = get_embedding(user_input, model, tokenizer)
    # st.write(f"Generated embedding: {embedding}")

    # Store in FAISS
    store_in_faiss(user_input, embedding, index, texts_file)
    save_faiss_index(index, index_file)

    # Load texts
    with open(texts_file, 'r') as f:
        texts = json.load(f)

    # Search for similar texts
    search_results = search_similar_texts_faiss(user_input, model, tokenizer, index, texts)
    st.write("Search Results similar to: " + user_input)
    for text, distance in search_results:
        st.write(f"Text: {text}, Distance: {distance}")

def main():
    load_dotenv()
    st.set_page_config(page_title="Text Embedding Storage", page_icon="📄")
    st.write("<style>body {background-color: #f0f2f6;}</style>", unsafe_allow_html=True)

    st.header("Text Embedding Storage 📄")

    user_input = st.text_input("Enter some text to store and search:")
    if st.button("Submit"):
        index_file = 'faiss_index.bin'
        texts_file = 'texts.json'
        index = load_faiss_index(index_file)
        handle_input(user_input, index, texts_file, index_file)

if __name__ == "__main__":
    main()
