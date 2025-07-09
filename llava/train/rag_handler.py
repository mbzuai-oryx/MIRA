import re
import faiss
import torch
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

def markdown_seq_check(text):
    # Define common markdown patterns
    markdown_patterns = [
        r"^#{1,6}\s",                # Headers (e.g. #, ##, ###)
        # r"\*\*.*?\*\*",               # Bold (e.g. **bold**)
        # r"\*.*?\*",                    # Italic (e.g. *italic*)
        # r"\[.*?\]\(.*?\)",             # Links (e.g. [text](url))
        r"^\s*[-*+]\s+",               # Unordered lists (- item, * item, + item)
        r"^\d+\.\s+",                   # Ordered lists (1. item)
        # r"`.*?`",                       # Inline code (e.g. `code`)
        r"```[\s\S]*?```",              # Code blocks (e.g. ```code```)
        r"^>.*",                        # Blockquotes (e.g. > quote)
        # r"_{2}.*?_{2}",                 # Underline (_italic_)
    ]
    
    markdown_regex = re.compile("|".join(markdown_patterns), re.MULTILINE)
    return bool(markdown_regex.search(text))

class query_rag_system():
    def __init__(self, tokenizer, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_csv(metadata_path)
        self.tokenizer = SentenceTransformer(tokenizer)
        
    @torch.no_grad()
    def retrieve(self, query, top_k=5):
        query_embedding = self.tokenizer.encode([query])

        # Search in the FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            paragraph = self.metadata.iloc[idx]['Paragraph']
            content = self.metadata.iloc[idx]['Content']
            results.append({
                'Paragraph': paragraph,
                'Content': content,
                'Distance': dist
            })

        return results

class query_rewrite_handler():
    def __init__(self, host_url = "http://10.127.104.16:11434/api/chat", target_model = "mistral-small:22b"):
        self.host_url = host_url
        self.target_model = target_model

    def rewrite(self, sources, question):
        paragraph = "You must only output modified question.\n[Chat history]\n"
        for m in sources:
            paragraph += m["from"] + ": " + m["value"] + "\n"
        
        paragraph += "[Now question]\n"
        paragraph += question["from"] + ": " + question["value"]

        prompt = [
            {"role": "system", "content": "Given [Chat history] and [Now question], please rewrite [Now question] to let it include information in [Chat history] that should be added into the question to make it more generally represent question intention and contain keywords it needs. You should only output modified [Now question]."},
            {"role": "user", "content": """You must only output modified question.
[Chat history]
human: <image>\nWhat is the view of the chest x-ray?
gpt: The chest x-ray is in the posteroanterior (P-A) view. This means that the x-ray beam passes from the back (posterior) of the patient to the front (anterior), with the image receptor placed in front of the patient.
human: What does the x-ray show about the mediastinum?
gpt: The x-ray shows a widening of the mediastinum. The mediastinum is the central compartment of the thoracic cavity, which contains various organs and structures such as the heart, major blood vessels, trachea, and esophagus. Widening of the mediastinum can be indicative of various conditions, such as inflammation, infection, or the presence of a mass.
[Now question]
human: Is there any abnormality in the neck region?"""},
            {"role": "assistant", "content": "Is there any abnormality in the neck region visible on the posteroanterior (P-A) chest x-ray, particularly in relation to the mediastinum?"},
            {"role": "user", "content": paragraph}
        ]
        results = requests.post(self.host_url, json={"model": self.target_model, "messages": prompt, "stream": False}).json()

        return results["message"]["content"]