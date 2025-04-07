def download_sample_text():
    # Tokenizing - short story by Edith Wharton, which has been released into the public domain
    # and is thus permitted to be used for LLM training tasks
    import urllib.request
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    filepath = "the-verdict.txt"
    urllib.request.urlretrieve(url, filepath)
    return filepath

def read_filepath(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data
    
if __name__ == "__main__":
    filepath = download_sample_text()
    print(f"filepath: {filepath}")