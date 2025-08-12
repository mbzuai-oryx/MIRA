import os
import re
import csv
import tqdm
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to download images
def download_image(url, folder_path, caption):
    global logger_file, error_file
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()  # Check if the request was successful

        # Extract the image name from the URL
        image_name = os.path.basename(url)
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the image locally
        image_path = os.path.join(folder_path, image_name)
        with open(image_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        caption_path = os.path.join(folder_path, ".".join(image_name.split(".")[:-1])+".txt")
        with open(caption_path, 'w') as cfile:
            cfile.write(caption)

        logger_file.write(f"Downloaded: {image_name}\n")

    except Exception as e:
        error_file.write(f"[Image Dl Fail]{url}: {e}\n")

# Function to get all image URLs from a Wikipedia page
def get_images_from_wikipedia(url):
    global error_file
    # Send a GET request to fetch the page content
    response = requests.get(url)
    if response.status_code != 200:
        error_file.write(f"[Image Ret Fail {response.status_code}]{url}\n")
        return None

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all images and captions
    images_with_captions = []
    for figure in soup.find_all('figure'):  # Search for <figure> tags which often contain images and captions
        img_tag = figure.find('img')  # Find the image tag within the figure
        caption_tag = figure.find('figcaption')  # Find the caption tag within the figure

        if img_tag:
            img_url = img_tag.get('src')  # Get the image source URL
            img_url = urljoin(url, img_url)  # Make the URL absolute

            caption = caption_tag.text.strip() if caption_tag else "No caption available"  # Get the caption text
            images_with_captions.append({'url': img_url, 'caption': caption})

    return images_with_captions

def get_page_content(url):
    # Send a GET request to fetch the page content
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve page: {url}")
        return None

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    category_dir = soup.find('ul', {'class': "vector-toc-contents"})
    categories = []

    def extract_category_name(s):
        # Regular expression to match the category name, allowing for multi-word names
        pattern = r"^\d+([A-Za-z\s]+?)(?=[A-Z0-9])"  # Capture after number, stopping before capital letter
        match = re.match(pattern, s)
        
        if match:
            # Extract the category name and strip any leading/trailing spaces
            category_name = match.group(1).strip()
            return category_name
        return None
    
    global waste_contents

    for li in category_dir.find_all('li'):
        text = li.get_text(strip=True)
        anchor = li.find('a')
        if re.search(r'\d', text): # has alphabetics
            nowcn = extract_category_name(text)
            if nowcn is not None:
                if nowcn.lower() not in waste_contents:
                    if anchor and 'href' in anchor.attrs:
                        # categories.append([nowcn, anchor["href"]])
                        categories.append(nowcn)
            else:
                if "." not in text:
                    nowcn = re.sub(r"[0-9]+", "", text)
                    if nowcn.lower() not in waste_contents:
                        if anchor and 'href' in anchor.attrs:
                            # categories.append([nowcn, anchor["href"]])
                            categories.append(nowcn)

    content_div = soup.find('div', {'class': 'mw-parser-output'})
    content_puretext_sp = content_div.get_text().split("\n")
    for mc in range(len(content_puretext_sp)):
        if "[edit]" in content_puretext_sp[mc]:
            content_puretext_sp[mc-1] += "<paragraph_sep>"
    content_puretext = "\n".join(content_puretext_sp)
    cpts = content_puretext.split("<paragraph_sep>")

    paragraph_sign = False
    category_paragraphs = [["starting", ""]]
    already_push_categories = []
    for m in cpts:
        if m.startswith("\n"):
            m = m.strip("\n")
        msep = m.replace("[edit]","").split("\n")

        if msep[0] in categories:
            paragraph_sign = True

        if not paragraph_sign:
            category_paragraphs[0][1] = category_paragraphs[0][1] + "\n".join(msep[1:])
        else:
            if msep[0] in categories and msep[0] not in already_push_categories:
                category_paragraphs.append([msep[0], "\n".join(msep[1:])])
                already_push_categories.append(msep[0])
            else:
                category_paragraphs[-1][1] += m.replace("[edit]","")

    return category_paragraphs

def API_summarization(paragraph, host_url, target_model):
    prompt = [
        {"role": "system", "content": "You are a helpful and professional medical scientist and is going to summarize and simplify the given medical instructions into a short and precise paragraph. Please remove citations and other layout characters. Only output ONE simplified paragraph without format."},
        {"role": "user", "content": "[Please only output one simplified paragraph.] Now summarize this:\n"+paragraph}
    ]
    results = requests.post(host_url, json={"model": target_model, "messages": prompt, "stream": False}).json()

    return results["message"]["content"]

# Main function
def crawl_wikipedia_page(wikipedia_url, download_folder, content_filename):
    try:
        if os.path.exists(content_filename): # Skip if the content file already exists
            return True

        image_cu = get_images_from_wikipedia(wikipedia_url)
        if image_cu is None:
            error_file.write("[Error Ext Im]{}\n".format(wikipedia_url))
        elif len(image_cu) == 0:
            error_file.write("[Error No Im]{}\n".format(wikipedia_url))
        else:
            for item in image_cu:
                if "https://upload.wikimedia.org/wikipedia/commons" in item["url"]:
                    download_image(item["url"], download_folder, caption = item["caption"])

        global ollama_api_dir, ollama_modeltype
        page_content = get_page_content(wikipedia_url)

        if page_content:
            if page_content==[["starting", ""]]:
                error_file.write("[Protected page]{}\n".format(wikipedia_url))
            else:
                # simplify
                for nowlist in page_content:
                    nowlist[1] = API_summarization(nowlist[1], ollama_api_dir, ollama_modeltype)

                # Save as csv
                with open(content_filename, mode="w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    page_content = [["Paragraph", "Content"]] + page_content
                    writer.writerows(page_content)
                logger_file.write("Successfully extracted {}\n".format(wikipedia_url))
        else:
            error_file.write("[Error Cnt]{}\n".format(wikipedia_url))
            return False
        
        return True
    except Exception as e:
        error_file.write("[System Error]{}: {}\n".format(wikipedia_url, e))
        return False

ollama_api_dir = "http://localhost:11434/api/chat"
ollama_modeltype = "qwen2.5:72b"
wikipedia_url_base = "https://en.wikipedia.org/wiki/"
waste_contents = ["see also", "references", "external links"]

# if os.path.exists("crawler_logs"):
#     import shutil
#     shutil.rmtree("crawler_logs")

os.makedirs("crawler_logs", exist_ok=True)
os.makedirs("./downloaded_pages/images", exist_ok=True)
os.makedirs("./downloaded_pages/text", exist_ok=True)

logger_filepath = "./crawler_logs/logs.txt"
error_filepath = "./crawler_logs/errors.txt"
logger_file = open(logger_filepath, "w+")
error_file = open(error_filepath, "w+")

max_threads = 16
failed = success = 0
task_list = []

disease_base = pd.read_csv('source/full_disease_dl.csv')
for index, row in disease_base.iterrows():
    if os.path.exists(row[2]):
        print(f"Skipping for existing: {row[2]}")
        continue
    task_list.append([row[0], row[1], row[2]])
    # crawl_wikipedia_page(row[0], row[1], row[2])

with ThreadPoolExecutor(max_threads) as executor:
    # Submit download tasks to the thread pool
    future_to_url = {executor.submit(crawl_wikipedia_page, tl[0], tl[1], tl[2]): tl for tl in task_list}
    
    for future in as_completed(future_to_url):
        if future.result() == False:
            failed += 1
        else:
            success += 1
        
        if (success + failed) % 100 == 0:
            print(f"Processed {success+failed}, Downloaded {success} files, {failed} failed")
        
        if (success + failed) % 10 == 0:
            logger_file.flush()
            error_file.flush()

logger_file.close()
error_file.close()