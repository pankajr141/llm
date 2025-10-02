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

    dataloader = create_dataloader(filepaths, batch_size=2, max_length=1024, stride=1024, drop_last=True, num_workers=0)

    total_count = 0
    for x, y in dataloader:
        # print(x.shape, y.shape)
        total_count += x.shape[0]
        if total_count % 100 == 0:
            print(f"Processed {total_count} samples so far...")

    print(f"Total samples in dataloader: {total_count}")
    print("Dataloader test passed.")

def test_instruction_finetune_dataloader():
    from llm_bhasa.harmony import data
    from llm_bhasa.harmony.dataset.dataset_instruction_instruction_finetune import create_dataloader

    base_urls = ["https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json", 
                "https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/refs/heads/main/data/alpaca_gpt4_data.json"]

    device = "cpu"
    dataloader = create_dataloader(base_urls, batch_size=2, max_length=1024, shuffle=True, device=device)

    total_count = 0
    for inputs, targets in dataloader:
        total_count += inputs.shape[0]
        if total_count % 100 == 0:
            print(f"Processed {total_count} samples so far...")

    print(f"Total samples in dataloader: {total_count}")
    print("Dataloader test passed.")

if __name__ == "__main__":
    test_instruction_finetune_dataloader()