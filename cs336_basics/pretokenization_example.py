# import os
# import re
# import pickle
# from typing import BinaryIO
# from collections import Counter
# import multiprocessing


# def find_chunk_boundaries(
#     file: BinaryIO,
#     desired_num_chunks: int,
#     split_special_token: bytes,
# ) -> list[int]:
#     """
#     Chunk the file into parts that can be counted independently.
#     May return fewer chunks if the boundaries end up overlapping.
#     """
#     assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

#     # Get total file size in bytes
#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     file.seek(0)

#     chunk_size = file_size // desired_num_chunks

#     # Initial guesses for chunk boundary locations, uniformly spaced
#     # Chunks start on previous index, don't include last index
#     chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
#     chunk_boundaries[-1] = file_size

#     mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

#     for bi in range(1, len(chunk_boundaries) - 1):
#         initial_position = chunk_boundaries[bi]
#         file.seek(initial_position)  # Start at boundary guess
#         while True:
#             mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

#             # If EOF, this boundary should be at the end of the file
#             if mini_chunk == b"":
#                 chunk_boundaries[bi] = file_size
#                 break

#             # Find the special token in the mini chunk
#             found_at = mini_chunk.find(split_special_token)
#             if found_at != -1:
#                 chunk_boundaries[bi] = initial_position + found_at
#                 break
#             initial_position += mini_chunk_size

#     # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
#     return sorted(set(chunk_boundaries))

# def pretokenize(filename, start, end, special_token):
#     #open file every process
#     with open(filename, "rb") as f:
#         f.seek(start)
#         chunk = f.read(end - start)

#     try:
#         chunk = chunk.decode("utf-8", errors="ignore")
#         special_token_escape = re.escape(special_token.decode("utf-8"))
#         chunk = re.split(special_token_escape, chunk)  # Remove trailing special token if present
#         chunk = "".join(chunk)
#         pair_counts = pretokenize_chunk(chunk)
#         return pair_counts
    
#     except UnicodeDecodeError:
#         print(f"UnicodeDecodeError at {start}-{end}, skipping chunk.")
#         return Counter()  # Return empty Counter on error

# def pretokenize_chunk(chunk: str) -> list[str]:
#     # Split on whitespace and punctuation
#     tokens = re.findall(r"\w+|[^\w\s]", chunk, re.UNICODE)
#     counts = Counter()
#     for token in tokens:
#         chars = list(token)
#         for i in range(len(chars) - 1):
#             counts.update([ (chars[i], chars[i+1]) ])
#     return counts

# ## Usage
# if __name__ == "__main__":
#     filename = "/home/wsl/cs336-assignment1-basics/tests/fixtures/tinystories_sample.txt"
#     #/home/wsl/cs336-assignment1-basics/tests/fixtures/tinystories_sample.txt
#     #/home/wsl/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt
#     num_processes = 16
#     special_token = b"<|endoftext|>"
#     with open(filename, "rb") as f:
#         boundaries = find_chunk_boundaries(f, num_processes, special_token)

#     tasks = []
#     for (start,end) in zip(boundaries[:-1], boundaries[1:]):
#         tasks.append((filename, start, end, special_token))
    
#     total_counts = Counter()
#     with multiprocessing.Pool(processes=num_processes) as pool:
#             results = pool.starmap(pretokenize, tasks)
#             for result in results:
#                 total_counts.update(result)
#     print(f"Total unique pairs: {len(total_counts)}")
#     print(total_counts.most_common(10))

#     # Optionally, save the results to a file
#     # results_filename = "pretokenization_counts.pkl"
#     # with open(results_filename, "wb")as f:
#     #     pickle.dump(total_counts,f)

import os
import re
import pickle
from typing import BinaryIO
from collections import Counter
import multiprocessing
import regex  # pip install regex


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pretokenize(filename, start, end, special_token):
    """Process a chunk and return word counts as byte tuples"""
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    try:
        # Decode to string
        text = chunk.decode("utf-8", errors="ignore")
        
        # Remove special tokens
        special_token_str = special_token.decode("utf-8")
        text = text.replace(special_token_str, "")
        
        # Count words using GPT-2 style regex
        word_counts = pretokenize_chunk(text)
        return word_counts
    
    except Exception as e:
        print(f"Error at {start}-{end}: {e}")
        return Counter()


def pretokenize_chunk(text: str) -> Counter:
    """
    Tokenize text using GPT-2 style pattern and return counts of byte tuples
    """
    # GPT-2 tokenization pattern
    pat = regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    
    tokens = regex.findall(pat, text)
    
    word_counts = Counter()
    for token in tokens:
        # Convert token to tuple of bytes
        token_bytes = tuple(bytes([b]) for b in token.encode('utf-8'))
        word_counts[token_bytes] += 1
    
    return word_counts


## Usage
if __name__ == "__main__":
    filename = "/home/wsl/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    num_processes = 8
    special_token = b"<|endoftext|>"
    
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token)

    tasks = []
    for (start, end) in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((filename, start, end, special_token))
    
    total_counts = Counter()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(pretokenize, tasks)
        for result in results:
            total_counts.update(result)
    
    print(f"Total unique tokens: {len(total_counts)}")
    print(total_counts.most_common(10))

    # Optionally, save the results to a file
    results_filename = "pretokenization_counts.pkl"
    with open(results_filename, "wb")as f:
        pickle.dump(total_counts,f)