import os
import glob
import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from rewritor import API_summarization, markdown_seq_check

# Configuration
csv_folder = "/home/jinhong.wang/workdir/database_rag/crawl_72bclean/downloaded_pages/text"  # Folder containing 1.csv to 5000.csv
vector_dim = 384  # Dimension of embeddings (depends on the model)
index_path = "/home/jinhong.wang/workdir/database_rag/rag_indexv2.1_0203.idx"  # Path to save the FAISS index
metadata_path = "/home/jinhong.wang/workdir/database_rag/rag_metadatav2.1_0203.csv"  # Path to save metadata (paragraphs and content)

# Load SentenceTransformer tokenizer
tokenizer = SentenceTransformer('all-MiniLM-L6-v2')

# tokenizer = transformers.AutoTokenizer.from_pretrained(
#             "lmsys/vicuna-7b-v1.5",
#             cache_dir=None,
#             model_max_length=2048,
#             padding_side="right",
#             use_fast=False,
#         )

if not os.path.exists(index_path):
    print("Generating new index...")
    # Step 1: Read and process CSV files
    all_paragraphs = []
    all_contents = []

    all_csv = glob.glob("/home/jinhong.wang/workdir/database_rag/crawler/downloaded_pages/text/*.csv")

    csv_done_count = 0
    for csv_path in tqdm.tqdm(all_csv):
        df = pd.read_csv(csv_path)

        disease_name = os.path.splitext(os.path.basename(csv_path))[0].replace("_"," ")
        pg = df['Paragraph'].tolist()
        for m in range(len(pg)):
            if pg[m] == "starting":
                pg[m] = disease_name
            else:
                pg[m] = disease_name + "'s " + pg[m]

        all_paragraphs.extend(pg)

        # Dealing with content rewrite
        nowcontent = df['Content'].tolist()
        for i in range(len(nowcontent)):
            max_retries = 0
            while markdown_seq_check(nowcontent[i]):
                nowcontent[i] = API_summarization(nowcontent[i], "http://localhost:11434/api/chat", "qwen2.5:32b")
                max_retries += 1
                if max_retries >= 3: break

        all_contents.extend(nowcontent)

        csv_done_count += 1
        if csv_done_count %10==0:
            mt2 = metadata_path.replace(".csv", "temp.csv")
            if os.path.exists(mt2):
                os.remove(mt2)
            pd.DataFrame({'Paragraph': all_paragraphs, 'Content': all_contents}).to_csv(mt2, index=False)

    pd.DataFrame({'Paragraph': all_paragraphs, 'Content': all_contents}).to_csv(metadata_path, index=False)
    data_to_index = [
        f"{p} {c}" for p, c in zip(all_paragraphs, all_contents) if isinstance(p, str) and isinstance(c, str)
    ]

    print("Indexing complete.")

    # Step 2: Generate embeddings
    print("Generating embeddings...")
    embeddings = tokenizer.encode(data_to_index, show_progress_bar=True)

    # Check embedding dimensions
    assert all(len(emb) == vector_dim for emb in embeddings), "Embedding dimensions do not match the index dimension."

    # Step 3: Create FAISS index
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(vector_dim)  # L2 distance metric
    index.add(embeddings)  # Add embeddings to the index

    # Save the index and metadata
    faiss.write_index(index, index_path)
    
else:
    print("Found index, load: ", index_path)
    index = faiss.read_index(index_path)

# Step 4: Querying the index
def query_rag_system(query, top_k=5):

    # import pdb; pdb.set_trace()
    # Generate embedding for the query
    query_embedding = tokenizer.encode([query])

    # Search in the FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Retrieve results
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        paragraph = metadata.iloc[idx]['Paragraph']
        content = metadata.iloc[idx]['Content']
        results.append({
            'Paragraph': paragraph,
            'Content': content,
            'Distance': dist
        })

    return results

# Example usage
query = "Finally, can you describe the fifth row for Corticobasal Degeneration (CBD)?"
results = query_rag_system(query)

print("Query: ", query)
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Paragraph: {result['Paragraph']}")
    print(f"Content: {result['Content']}")
    print(f"Distance: {result['Distance']}")
    print("-")
