import re, requests

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

def API_summarization(paragraph, host_url, target_model):
    prompt = [
        {"role": "system", "content": "Please simpilify user's given text into one paragraph. Please keep necessary information from the original paragraph so that reader can clearly understand what the paragraph is about. Please describe from an encyclopedic perspective, i.e. use third person perspective. Only output one clear and informative paragraph with no format and no citations. Do not make it like markdown, make it ONE PARAGRAPH."},
        {"role": "user", "content": paragraph}
    ]
    results = requests.post(host_url, json={"model": target_model, "messages": prompt, "stream": False}).json()

    return results["message"]["content"]