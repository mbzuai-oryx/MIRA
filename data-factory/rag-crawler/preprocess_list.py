import pandas as pd
import tqdm

disease_base = pd.read_csv('human_disease_textmining_full.tsv', sep='\t', header=None)
failed = success = 0
wikipedia_url_base = "https://en.wikipedia.org/wiki/"

task_list = []
dl_dup_set = []

url = []
imdir = []
tdir = []

for index, row in tqdm.tqdm(disease_base.iterrows(), total = len(disease_base)):
    now_disease = row[3].replace(" ", "_")
    if now_disease not in dl_dup_set:
        dl_dup_set.append(now_disease)
    else:
        continue

    wikipedia_url = wikipedia_url_base + now_disease
    url.append(wikipedia_url)
    imdir.append("./downloaded_pages/images/"+now_disease)
    tdir.append("./downloaded_pages/text/"+now_disease+".csv")

df = pd.DataFrame({"url": url, "image_dir": imdir, "text_dir": tdir})
df.to_csv("full_disease_dl.csv", index=False, encoding='utf-8')
    