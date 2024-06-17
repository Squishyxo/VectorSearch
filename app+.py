import streamlit as st
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5Model
from elasticsearch import Elasticsearch
import os
import torch

# Load environment variables
def load_dotenv():
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()

# Get embedding for a given text
def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    decoder_input_ids = tokenizer(" ", return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Store text and embedding in Elasticsearch
def store_in_elasticsearch(text, embedding):
    es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))
    doc = {
        "text": text,
        "embedding": embedding
    }
    res = es.index(index="text-embeddings", body=doc)
    return res

# Search for similar texts in Elasticsearch
def search_similar_texts(user_input, model, tokenizer):
    es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))
    embedding = get_embedding(user_input, model, tokenizer)
    
    query = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {
                        "query_vector": embedding
                    }
                }
            }
        }
    }
    
    try:
        response = es.search(index="text-embeddin", body=query)
    except Exception as e:
        st.error(f"Error: {e}")
        response = None
    return response

# Display search results to the user
def display_results(results):
    if results:
        hits = results['hits']['hits']
        for hit in hits:
            st.write(f"Score: {hit['_score']}, Text: {hit['_source']['text']}")

# Main Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Text Embedding Storage and Search", page_icon="📄")
    st.write("<style>body {background-color: #f0f2f6;}</style>", unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Text Embedding Storage and Search 📄")

    user_input = st.text_input("Enter some text:")
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

    # Search for similar texts
    st.write("Searching for similar texts...")
    search_results = search_similar_texts(user_input, model, tokenizer)
    display_results(search_results)

if __name__ == "__main__":
    main()
