from collections import Counter, defaultdict
import regex as re
import heapq
import os
import pickle
import json

def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(i) for i in cs]
    return dict(zip(bs, cs))

def merge_pair(
    word: tuple[bytes, ...], 
    pair: tuple[bytes, bytes], 
    new_token: bytes
) -> tuple[bytes, ...]:
    new_word = []
    i =0
    while i < len(word):
        # check if merge pair
        if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
            new_word.append(new_token)
            # skip two addresses
            i += 2
        else:
            new_word.append(word[i])
            # skip one address
            i += 1
    return tuple(new_word)

def pretokenization(text: str, special_tokens: list[str] = None, pkl_input_path: str = None) -> Counter:
    # pretokenize
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # split text by special tokens
    if special_tokens:
        special_pattern = '|'.join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        text_parts = re.split(f'({special_pattern})', text)
    else:
        text_parts = [text]
    
    if pkl_input_path and os.path.exists(pkl_input_path):
        with open(pkl_input_path, 'rb') as f:
            word_counts = pickle.load(f)
        return word_counts

    # get word counts
    word_counts = Counter()
    for part in text_parts:
        if part in special_tokens:
            word_counts[tuple([part.encode('utf-8')])] += 1
        elif part:
            tokens = re.findall(pat, part)
            for token in tokens:
                token_bytes = token.encode('utf-8')
                word_counts[tuple(bytes([b]) for b in token_bytes)] += 1

    return word_counts

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    pkl_input_path: str = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Read file
    with open(input_path, 'r', encoding='utf-8')as f:
        text = f.read()
    
    # Initial vocab
    # 256 bytes
    vocab  = {}
    for i in range(256):
        vocab[i] = bytes([i])

    # add special_tokens
    for special_token in special_tokens:
        if special_token.encode('utf-8') not in vocab.values():
            vocab[len(vocab)] = special_token.encode('utf-8')

    word_counts = pretokenization(text, special_tokens, pkl_input_path)

    # # pretokenize
    # pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # # split text by special tokens
    # if special_tokens:
    #     special_pattern = '|'.join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    #     text_parts = re.split(f'({special_pattern})', text)
    # else:
    #     text_parts = [text]
    
    # # get word counts
    # word_counts = Counter()
    # for part in text_parts:
    #     if part in special_tokens:
    #         word_counts[tuple([part.encode('utf-8')])] += 1
    #     elif part:
    #         tokens = re.findall(pat, part)
    #         for token in tokens:
    #             token_bytes = token.encode('utf-8')
    #             word_counts[tuple(bytes([b]) for b in token_bytes)] += 1

    # BPE training
    merges = []

    # set of special token bytes
    special_token_bytes = {st.encode('utf-8') for st in special_tokens}

    while len(vocab) < vocab_size:
        # Count all pairs
        pair_counts = Counter()
        
        for word, count in word_counts.items():
            # Skip special tokens (single-element tuples containing special token bytes)
            if len(word) == 1 and word[0] in special_token_bytes:
                continue
            
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += count
        
        if not pair_counts:
            break
        
        # Find best pair: max count, with lexicographic tiebreaker
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        
        # Merge new best pair and add to vocab
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)
        
        # Update word_counts
        new_word_counts = Counter()
        
        for word, count in word_counts.items():
            # Skip special tokens
            if len(word) == 1 and word[0] in special_token_bytes:
                new_word_counts[word] += count
                continue
            
            # Check if word contains best_pair
            has_best_pair = False
            for i in range(len(word) - 1):
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    has_best_pair = True
                    break
            
            if has_best_pair:
                # Merge the pair in this word
                new_word = merge_pair(word, best_pair, new_token)
                new_word_counts[new_word] += count
            else:
                new_word_counts[word] += count
        
        word_counts = new_word_counts

    return vocab, merges

def save_bpe_results(vocab, merges, base_output_path="bpe_output"):
    # 1. 初始化 GPT-2 字节编码器
    byte_encoder = gpt2_bytes_to_unicode()

    # 2. 保存词汇表 (vocab.json): token_string -> token_id
    vocab_output_path = f"{base_output_path}.json"
    token_to_id = {}
    
    for token_id, token_bytes in vocab.items():
        # 将字节 token 转换为 GPT-2 专用的 Unicode 字符串
        gpt2_tokens = "".join([byte_encoder[b] for b in token_bytes])
        token_to_id[gpt2_tokens] = token_id

    try:
        with open(vocab_output_path, 'w', encoding='utf-8') as f:
            json.dump(token_to_id, f, indent=4, ensure_ascii=False)
        print(f"Vocab saved successfully to {vocab_output_path}")
    except Exception as e:
        print(f"Error saving vocab to JSON: {e}")

    # 3. 保存合并规则 (merges.txt): token1 token2
    merges_output_path = f"{base_output_path}.txt"
    
    try:
        with open(merges_output_path, 'w', encoding='utf-8') as f:

            f.write("#version: 1.0\n") 
            
            for token_a_bytes, token_b_bytes in merges:
                token_a_str = "".join([byte_encoder[b] for b in token_a_bytes])
                token_b_str = "".join([byte_encoder[b] for b in token_b_bytes])
                
                f.write(f"{token_a_str} {token_b_str}\n")
                
        print(f"Merges saved successfully to {merges_output_path}")
    except Exception as e:
        print(f"Error saving merges to TXT: {e}")


if __name__ == "__main__":
    input_path = "/home/wsl/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    pkl_input_path = "/home/wsl/cs336-assignment1-basics/data/pretokenization_counts_1.pkl"
    vocab, merges = run_train_bpe(
        input_path = input_path,
        vocab_size = vocab_size,
        special_tokens = special_tokens,
        pkl_input_path = pkl_input_path,
    )

    SAVE_FILE_BASE = "bpe_output_1"
    save_bpe_results(vocab, merges, SAVE_FILE_BASE)
    
    print("Vocabulary:")
    for token_id, token_bytes in list(vocab.items())[:10]:
        try:
            display_bytes = token_bytes.decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            display_bytes = str(token_bytes)
        print(f"{token_id}: {display_bytes}")

    print("\nMerges:")
    for merge in merges[:10]:
        print(f"{merge[0]} {merge[1]}")