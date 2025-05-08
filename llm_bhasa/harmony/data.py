import os
import re
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import urllib.request

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

def _download_gutenberg_books(args):
    """Downloads a single book from Project Gutenberg.

    This is a helper function used internally by `download_gutenberg_books`.
    It attempts to download a book given its ID and a download directory.

    Args:
        args (tuple): A tuple containing the book ID (int) and the download directory (str).

    Returns:
        tuple: A tuple containing a boolean indicating success (True) or failure (False),
               and either the filepath of the downloaded book (on success) or an error message (on failure).
    """
    book_id, download_dir = args
    base_url = "https://www.gutenberg.org/files/{}/{}-0.txt"
    url = base_url.format(book_id, book_id)
    filepath = os.path.join(download_dir, f"{book_id}.txt")
    try:
        urllib.request.urlretrieve(url, filepath)
    except urllib.error.HTTPError as e:
        return False, f"Error downloading book ID {book_id}: {e}"
    except Exception as e:
        return False, f"An unexpected error occurred while downloading book ID {book_id}: {e}"
    return True, filepath

def download_gutenberg_books(book_ids, download_dir="gutenberg_books", verbose=True, njobs=1):
    """Downloads multiple books from Project Gutenberg.

    This function downloads a collection of books from Project Gutenberg, given their IDs.
    It supports both single-threaded and multi-threaded downloading for efficiency.

    Args:
        book_ids (list): A list of Project Gutenberg book IDs (integers) to download.
        download_dir (str, optional): The directory where the downloaded books will be saved.
            Defaults to "gutenberg_books".
        verbose (bool, optional): A flag indicating whether to print download progress and error messages.
            Defaults to True.
        njobs (int, optional): The number of jobs to run in parallel for downloading.
            If 1, downloads are performed sequentially. If greater than 1, downloads are multi-threaded.
            Defaults to 1.

    Returns:
        list: A list of filepaths of the successfully downloaded books.
    """
    filepaths = []

    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    if njobs == 1:
        success = 0
        failure = 0
        with tqdm(book_ids, desc="Downloading gutenberg books", postfix={"success": success, "failure": failure}) as pbar:
            for book_id in pbar:
                status, filepath = _download_gutenberg_books((book_id, download_dir))
                if status:
                    filepaths.append(filepath)
                    success += 1
                else:
                    failure += 1
                    print_on_verbose(filepath, verbose)
                    
                pbar.set_postfix({"success": success, "failure": failure})
    else:
        args = list(zip(book_ids, [download_dir] * len(book_ids)))
        result = process_map(_download_gutenberg_books, 
                             args,              
                             max_workers=njobs,
                             chunksize=1,
                             desc=f"Downloading gutenberg books")
        filepaths = list(filter(lambda x: x[0], result))
    return filepaths

def download_sample_text(gutenberg_book_ids=[1342, 84, 1661], verbose=True, njobs=1):
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
        njobs (int, optional): The number of jobs to run in parallel for downloading.
            If 1, downloads are performed sequentially. If greater than 1, downloads are multi-threaded.
            Defaults to 1.

    Returns:
        list: A list of filepaths where the downloaded text file and Gutenberg
            books are saved.
    """
    import urllib.request
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    filepath = "the-verdict.txt"
    urllib.request.urlretrieve(url, filepath)

    # Download multiple books from Project Gutenberg
    gutenberg_filepaths = download_gutenberg_books(gutenberg_book_ids, verbose=verbose, njobs=njobs)

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

    gutenberg_book_ids = range(10)
    filepaths = download_sample_text(gutenberg_book_ids=gutenberg_book_ids, verbose=False, njobs=1)
    print(f"filepaths: {len(filepaths)}")
    