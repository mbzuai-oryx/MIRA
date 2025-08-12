import json, os, shutil, tqdm

import requests
from urllib.parse import urlparse
import os
from PIL import Image
from io import BytesIO

def download_image(image_url, save_dir):
    """
    Downloads an image from a URL and saves it to the specified directory.
    
    Args:
        image_url (str): The URL of the image to download
        save_dir (str): The directory to save the image to
        
    Returns:
        bool: True if the download was successful, False otherwise
    """
    try:
        # Get the image filename from the URL
        parsed_url = urlparse(image_url)
        filename = os.path.basename(parsed_url.path)
        
        # If the filename is empty or doesn't have an extension, use a default name
        if not filename or '.' not in filename:
            filename = 'image.jpg'
            
        # Full path to save the image
        save_path = os.path.join(save_dir, filename)
        
        # Define headers with a User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        # Download the image with headers
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        # Verify the image can be opened (valid image file)
        Image.open(BytesIO(response.content))
        
        return True
        
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        return False

for i in tqdm.tqdm(range(500)):
    if not os.path.exists(f"rewritten_data_rag2/{i}.json"):
        print(f"rewritten_data_rag2/{i}.json does not exist")
        continue
    os.makedirs(f"rag_for_selection/{i}", exist_ok=True)
    nowset = json.load(open(f"rewritten_data_rag2/{i}.json"))
    nowset_tosave = {}
    with open(f"rag_for_selection/{i}/{i}.txt", "w") as f:
        # RAGs

        shutil.copy("./WashMRAG/it_imfiles/"+nowset["image"], f"rag_for_selection/{i}/input_image.jpg")
        shutil.copy(f"rewritten_data_rag2/{i}.json", f"rag_for_selection/{i}/raw_input_data.json")

        nowrag = nowset["rag"]
        nowrag_ims = nowset["rag_img"]
        nowrag_online = nowset["rag_online"][0]
        nowrag_online_ims = nowset["rag_online_im"]

        rag_sequence = ""
        for m in nowrag:
            if "certainly!" in m["Content"].lower():
                continue
            rag_sequence+= m["Paragraph"] + ": " + m["Content"] + "\n"

        if nowrag_online is not None and nowrag_online != "":
            rag_sequence += "Online RAG:" + nowrag_online + "\n"

        nowidx = 2
        for m in nowrag_ims:
            # Check if the image description is NaN (not a number)
            if not isinstance(m["Description"], str):
                # Skip this image or use a placeholder description
                m["Description"] = "(No description available)"

            rag_sequence+= "[Image " + str(nowidx) + "] which content is: " + m["Description"] + "\n"
            
            shutil.copy(
                os.path.join("./WashMRAG/rag_imfiles", m["Image_File"]), 
                os.path.join(f"rag_for_selection/{i}", f"[RAGImage{str(nowidx)}]"  + os.path.basename(m["Image_File"]))
                )
            
            nowidx+=1
            
        for m in nowrag_online_ims:
            rag_sequence+= "[Image " + str(nowidx) + "] which content is: " + m["title"] + "\n"
            with open(f"rag_for_selection/{i}/[RAGOnlineImage{str(nowidx)}].txt", "w") as f2:
                f2.write(m["url"])

            # file_extension = "."+ m["url"].split(".")[-1]
            # if not file_extension or file_extension.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            #     with open(f"rag_for_selection/{i}/[RAGOnlineImage{str(nowidx)}].txt", "w") as f:
            #         f.write(m["url"])
            # else:
            #     file_extension = "."+ m["url"].split(".")[-1]
            # download_image(m["url"], f"rag_for_selection/{i}/[RAGOnlineImage{str(nowidx)}]" + file_extension)

            nowidx+=1

        f.write("(*)[Conversation]:\n")
        for conv in nowset["conversations"]:
            f.write(conv["from"] + ": " + conv["value"] + "\n")

        f.write("\n(*)[RAG Sequence after each query]:\n")
        f.write(rag_sequence)

        