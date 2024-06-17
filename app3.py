import os
from elasticsearch import Elasticsearch
from transformers import T5Tokenizer, T5Model
import torch

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Elasticsearch
es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))

# Initialize T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5Model.from_pretrained('t5-small')

# Function to get embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    decoder_input_ids = tokenizer(" ", return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Index a sample document
text = "Sample text for embedding"
embedding = get_embedding(text)
doc = {
    "text": text,
    "embedding": embedding
}

response = es.index(index="text-embeddings", body=doc)
print(response)
