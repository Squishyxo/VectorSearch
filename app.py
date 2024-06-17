import streamlit as st
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5Model
from elasticsearch import Elasticsearch
import os
import torch

def load_dotenv():
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    decoder_input_ids = tokenizer(" ", return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def store_in_elasticsearch(text, embedding):
    es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))
    doc = {
        "text": text,
        "embedding": embedding
    }
    res = es.index(index="text-embeddings", body=doc)
    return res

def main():
    load_dotenv()
    st.set_page_config(page_title="Text Embedding Storage", page_icon="📄")
    st.write("<style>body {background-color: #f0f2f6;}</style>", unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Text Embedding Storage 📄")

    user_input = st.text_input("Enter some text to store:")
    if st.button("Submit"):
        handle_input(user_input)


def handle_input(user_input):
    st.write("Processing your input...")
    # Load the model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5Model.from_pretrained('t5-small')

    # Get the embedding
    embedding = get_embedding(user_input, model, tokenizer)
    st.write(f"Generated embedding: {embedding}")

    # Store in Elasticsearch
    result = store_in_elasticsearch(user_input, embedding)
    st.write(f"Stored in Elasticsearch: {result}")

if __name__ == "__main__":
    main()
