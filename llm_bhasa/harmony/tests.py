import os
import sys
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
print(path)
sys.path.append(path)

def test_dataloader():
    from llm_bhasa.harmony import data
    from llm_bhasa.harmony.dataset.dataset_llm import create_dataloader


    gutenberg_book_ids = range(9)  # 100

    # Download file in localstorage
    # filepaths = data.download_sample_text(gutenberg_book_ids=gutenberg_book_ids, verbose=False)

    # Direct download in memory
    base_url = "https://www.gutenberg.org/files/{}/{}-0.txt"
    filepaths = [base_url.format(book_id, book_id) for book_id in gutenberg_book_ids]

    dataloader = create_dataloader(filepaths, batch_size=1, max_length=256, shuffle=True, chunk_size=2, stride=1)
    for x, y in dataloader:
        print(x.shape, y.shape)
        break

if __name__ == "__main__":
    test_dataloader()