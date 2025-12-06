from typing import Iterable, Iterator
import json
import regex as re

pat = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab

        self.vocab_reverse = {v:k for k,v in vocab.items()}

        self.merge_rules = {pair: idx for idx, pair in enumerate(merges)}

        self.special_tokens = special_tokens or []
        self.special_tokens_set = set(self.special_tokens)

        self.pat = re.compile(pat)

        self.special_token_encoder = {}
        for token_str in self.special_tokens:
            token_bytes = token_str.encode("utf-8")
            if token_bytes in self.vocab_reverse:
                self.special_token_encoder[token_str] = self.vocab_reverse[token_bytes]
            
    # @classmethod
    # def from_files(
    #     cls,
    #     vocab_file: str,
    #     merges_file: str,
    #     special_tokens: list[str] | None = None,
    # ) -> "Tokenizer":
    #     with open(vocab_file, "r", encoding="utf-8") as vf:
    #         vocab_data = json.load(vf)
    #         vocab = {int(k): v.encode("utf-8") for k, v in vocab_data.items()}

    #     merges = []
    #     with open(merges_file, "r", encoding="utf-8") as mf:
    #         for line in mf:
    #             line = line.strip()
    #             if not line or line.startswith("#"):
    #                 continue
    #             parts = line.split()
    #             if len(parts) != 2:
    #                 raise ValueError(f"Invalid merge line: '{line}'")
    #             merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

    #     return cls(vocab, merges, special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_file: str,
        merges_file: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        # --- 1. 构建逆向解码器 ---
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(i) for i in cs]
        decoder = {v: k for k, v in zip(bs, cs)}

        def decode_token_str(token_str: str) -> bytes:
            return bytes([decoder[c] for c in token_str])

        # --- 2. 加载 Vocab (修正了这里！) ---
        with open(vocab_file, "r", encoding="utf-8") as vf:
            vocab_data = json.load(vf)
            # 你的 JSON 是 {"Token": ID} 格式
            # k 是 Token 字符串 ("Ā"), v 是 ID (0)
            # 我们需要把它变成 {ID: Bytes} 格式
            vocab = {}
            for token_str, token_id in vocab_data.items():
                vocab[token_id] = decode_token_str(token_str)

        # --- 3. 加载 Merges ---
        merges = []
        with open(merges_file, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue 
                merges.append((decode_token_str(parts[0]), decode_token_str(parts[1])))

        return cls(vocab, merges, special_tokens)

    def _merge_tokens(self, tokens: list[bytes]) -> list[bytes]:
        while len(tokens) >= 2:
            merge_candidates = [
                (i, (tokens[i], tokens[i + 1]))
                for i in range(len(tokens) - 1)
                if (tokens[i], tokens[i + 1]) in self.merge_rules
            ]
            if not merge_candidates:
                break
            
            # Find the best pair to merge based on the merge rules
            best_idx, best_pair = min(
                merge_candidates,
                key=lambda x: self.merge_rules[x[1]]
            )

            merged_token = best_pair[0]+ best_pair[1]
            tokens = (
                tokens[:best_idx] +
                [merged_token] +
                tokens[best_idx + 2:]
            )
        return tokens
    
    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            token_ids = []
            for part in re.findall(self.pat, text):
                text_bytes = text.encode('utf-8')

                tokens = [bytes([b]) for b in text_bytes]
                merged_tokens = self._merge_tokens(tokens)
                # Convert merged tokens to their corresponding IDs
                token_ids = [self.vocab_reverse[token] for token in merged_tokens]
            return token_ids
        
        sorted_special = sorted(self.special_tokens,key=len, reverse=True )
        pattern = "(" + "|".join(re.escape(t) for t in sorted_special) + ")"
        parts = re.split(pattern, text)
        token_ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens_set:
                token_ids.append(self.special_token_encoder[part])
            else:
                regex_match = re.findall(self.pat, part)
                for part in regex_match:
                    text_bytes = part.encode('utf-8')
                    if text_bytes:
                        tokens = [bytes([b]) for b in text_bytes]
                        merged_tokens = self._merge_tokens(tokens)
                        # Convert merged tokens to their corresponding IDs
                        part_token_ids = [self.vocab_reverse[token] for token in merged_tokens]
                        token_ids.extend(part_token_ids)
        return token_ids
    
    def encode_iterable(self, iterable:Iterable[str]) -> Iterator[list[int]]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[id_] for id_ in ids]
        decode_bytes = b"".join(bytes_list)
        return decode_bytes.decode("utf-8", errors="replace")