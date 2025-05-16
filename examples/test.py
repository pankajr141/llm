from llm_bhasa.harmony import dataset, data
gutenberg_book_ids = range(9)  # 100

# Download file in localstorage
filepaths = data.download_sample_text(gutenberg_book_ids=gutenberg_book_ids, verbose=False)

# Direct download in memory
base_url = "https://www.gutenberg.org/files/{}/{}-0.txt"
filepaths = [base_url.format(book_id, book_id) for book_id in gutenberg_book_ids]

dataloader = dataset.create_dataloader(filepaths, batch_size=1, max_length=4, stride=1)
for i, batch in enumerate(dataloader):
    if i % 1000 == 0:
        print(i)
print(i)