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

# Additional sample texts to index
additional_texts = [
    "I enjoy listening to classical music while reading.",
    "She loves growing her own vegetables in the backyard.",
    "He likes to study the night sky with his telescope.",
    "The autumn leaves create a picturesque scene in the park.",
    "I love experimenting with new cuisines in the kitchen.",
    "The library hosts weekly book club meetings.",
    "She enjoys mountain climbing during the summer.",
    "He loves taking his dog for long walks in the woods.",
    "I like to visit historic battlefields and monuments.",
    "The botanical garden has a stunning butterfly exhibit.",
    "She enjoys making quilts from old clothes.",
    "He likes to practice archery in his free time.",
    "I love snorkeling in coral reefs.",
    "The beach is a great place for a morning jog.",
    "She enjoys writing and illustrating children's books.",
    "He likes to brew different types of tea.",
    "I enjoy attending art festivals and craft fairs.",
    "The lake is a popular spot for ice fishing in winter.",
    "She loves making her own herbal remedies.",
    "He enjoys taking landscape photos during his travels.",
    "I like to visit local farmer's markets on weekends.",
    "The theater offers a summer camp for young actors.",
    "She enjoys learning about ancient mythologies.",
    "He loves to restore vintage cars in his garage.",
    "I like to read science journals and articles.",
    "The park has a dedicated area for flying kites.",
    "She enjoys making stained glass art.",
    "He likes to go on nature walks to observe wildlife.",
    "I love attending food and wine tasting events.",
    "The river is a popular location for bird watching."
]

# Index each additional text
for text in additional_texts:
    embedding = get_embedding(text)
    doc = {
        "text": text,
        "embedding": embedding
    }
    response = es.index(index="text-embeddings", body=doc)
    print(f"Indexed document: {response['result']} - {text}")
