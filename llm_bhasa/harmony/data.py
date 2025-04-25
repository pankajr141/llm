import urllib.request
import os
import re

def print_on_verbose(text, verbose):
    """
    Prints text to the console if verbose mode is enabled.

    This function provides a way to conditionally print information during
    program execution. If the `verbose` flag is set to True, the provided
    text will be printed to the console. Otherwise, no output will be
    generated.

    Args:
        text (str): The text to be printed.
        verbose (bool): A flag indicating whether to print the text.
            If True, the text will be printed. If False, nothing will be printed.

    Returns:
        None
    """
    if not verbose:
        return
    print(text)

def download_gutenberg_books(book_ids, download_dir="gutenberg_books", verbose=True):
    """
    Downloads multiple books from Project Gutenberg.

    This function downloads books from Project Gutenberg based on their book IDs.
    It handles potential errors during download and saves each book as a separate
    text file in the specified directory.

    Args:
        book_ids (list): A list of Project Gutenberg book IDs (integers).
        download_dir (str, optional): The directory to save the downloaded books.
            Defaults to "gutenberg_books".
        verbose (bool, optional): A flag indicating whether to print download
            progress. Defaults to True.

    Returns:
        list: A list of filepaths where the downloaded books are saved.
            Returns an empty list if no books were successfully downloaded.
    """
    base_url = "https://www.gutenberg.org/files/{}/{}-0.txt"
    filepaths = []

    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    for book_id in book_ids:
        url = base_url.format(book_id, book_id)
        filepath = os.path.join(download_dir, f"{book_id}.txt")

        try:
            print_on_verbose(f"Downloading book ID {book_id} from {url}...", verbose)
            urllib.request.urlretrieve(url, filepath)
            filepaths.append(filepath)
            print_on_verbose(f"Successfully downloaded book ID {book_id} to {filepath}", verbose)
        except urllib.error.HTTPError as e:
            print_on_verbose(f"Error downloading book ID {book_id}: {e}", verbose)
        except Exception as e:
            print_on_verbose(f"An unexpected error occurred while downloading book ID {book_id}: {e}", verbose)

    return filepaths

def download_sample_text(gutenberg_book_ids=[1342, 84, 1661], verbose=True):
    """
    Downloads a sample text file and multiple books from Project Gutenberg.

    This function retrieves a text file containing a short story by Edith Wharton,
    which is in the public domain, making it suitable for LLM training tasks.
    The downloaded file is saved locally as 'the-verdict.txt'.
    It also downloads multiple books from Project Gutenberg based on the provided
    book IDs and saves them in a separate directory ('gutenberg_books').

    Args:
        gutenberg_book_ids (list, optional): A list of Project Gutenberg book IDs
            to download. Defaults to [1342, 84, 1661] (Pride and Prejudice,
            Frankenstein, The Adventures of Sherlock Holmes).
        verbose (bool, optional): A flag indicating whether to print download
            progress. Defaults to True.
    Returns:
        list: A list of filepaths where the downloaded text file and Gutenberg
            books are saved.
    """
    import urllib.request
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    filepath = "the-verdict.txt"
    urllib.request.urlretrieve(url, filepath)

    # Download multiple books from Project Gutenberg
    gutenberg_filepaths = download_gutenberg_books(gutenberg_book_ids, verbose=verbose)

    return [filepath] + gutenberg_filepaths

def read_filepaths(filepaths):
    """
    Reads and returns the concatenated content of multiple text files.

    This function takes a list of filepaths, opens each file in read mode with
    UTF-8 encoding, reads the entire content, and concatenates the content
    from all files into a single string.

    Args:
        filepaths (list): A list of filepaths to be read.

    Returns:
        str: The concatenated content of all files as a single string.
    """
    concatenated_text = ""
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as file:
            concatenated_text += file.read()
    return concatenated_text

if __name__ == "__main__":

    gutenberg_book_ids = range(100)
    filepaths = download_sample_text(gutenberg_book_ids=gutenberg_book_ids, verbose=True)
    textdata = read_filepaths(filepaths)
    print(f"text: {textdata[:100]}")
