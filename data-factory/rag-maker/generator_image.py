import os
import glob
import tqdm
import pandas as pd
import numpy as np
import faiss
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, SiglipModel, SiglipProcessor

# Configuration
image_folder = "/home/jinhong.wang/workdir/database_rag/crawl_72bclean/downloaded_pages/images"  # Folder containing image and description pairs
index_path = "ragim_index_0203.idx"  # Path to save the FAISS index
metadata_path = "ragim_metadata_0203.csv"  # Path to save metadata
vector_dim = 1152

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SigLIP model and processor
model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384", ignore_mismatched_sizes=True).to(device)
processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# Check if FAISS index exists; if not, generate it
if not os.path.exists(index_path):
    print("Generating new index...")
    # Collect all image files
    # Collect all image files with common extensions (png, jpg, jpeg)
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, "**", ext), recursive=True))
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} image files")

    embeddings = []
    descriptions = []
    # Process each image
    for image_file in tqdm.tqdm(image_files):
        # Load image
        image = Image.open(image_file).convert("RGB")
        # Preprocess image for CLIP
        inputs = processor(images=image, return_tensors="pt")
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Generate embedding
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # Normalize embedding for cosine similarity
        embedding = image_features / image_features.norm(dim=-1, keepdim=True)
        embedding = embedding.squeeze().cpu().numpy()
        embeddings.append(embedding)
        # Load corresponding description
        description_file = os.path.splitext(image_file)[0] + ".txt"
        with open(description_file, "r") as f:
            description = f.read()
        descriptions.append(description)

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings, dtype=np.float32)

    # Create FAISS index with inner product (cosine similarity for normalized vectors)
    # Get the actual dimension of the embeddings from the first embedding
    # actual_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(vector_dim)
    # print(f"Creating index with dimension: {actual_dim}")
    index.add(embeddings)

    # Save the index
    faiss.write_index(index, index_path)

    # Create and save metadata
    metadata = pd.DataFrame({
        'Image_File': image_files,
        'Description': descriptions
    })
    metadata.to_csv(metadata_path, index=False)
    print("Index and metadata saved.")
else:
    print(f"Found existing index, loading: {index_path}")
    index = faiss.read_index(index_path)

# Query function to retrieve similar images
def query_rag_system(query_image_path, top_k=5):
    """
    Retrieve the top-k most similar images and their descriptions based on a query image.
    
    Args:
        query_image_path (str): Path to the query image.
        top_k (int): Number of similar images to retrieve (default: 5).
    
    Returns:
        list: List of dictionaries containing image file paths, descriptions, and similarity scores.
    """
    # Load and preprocess query image
    query_image = Image.open(query_image_path)
    inputs = processor(images=query_image, return_tensors="pt")
    # Move inputs to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate embedding for query image
    with torch.no_grad():
        query_features = model.get_image_features(**inputs)
    # Normalize embedding
    query_embedding = query_features / query_features.norm(dim=-1, keepdim=True)
    query_embedding = query_embedding.squeeze().cpu().numpy().reshape(1, -1)

    # Search FAISS index for top-k similar images
    distances, indices = index.search(query_embedding, top_k)

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Collect results
    results = []
    for idx, sim in zip(indices[0], distances[0]):
        result = metadata.iloc[idx]
        results.append({
            'Image_File': result['Image_File'],
            'Description': result['Description'],
            'Similarity': sim
        })
    return results

# Example usage
if __name__ == "__main__":
    query_image_path = "/home/jinhong.wang/workdir/database_rag/testimg.jpg"  # Replace with actual query image path
    results = query_rag_system(query_image_path, top_k=1)

    print(f"Query Image: {query_image_path}")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Image: {result['Image_File']}")
        print(f"Description: {result['Description']}")
        print(f"Similarity: {result['Similarity']}")