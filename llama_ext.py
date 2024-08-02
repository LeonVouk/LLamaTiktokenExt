import os
import json
import regex as re
import time
import gc
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer

from ext_llama import train_loop as rust_train_loop

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LLAMA3_SPLIT_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

class Tokenizer:
    def __init__(self):
        self.merges = []
        self.pattern = LLAMA3_SPLIT_PATTERN
        self.special_tokens = {}
        self.vocab = self._build_vocab()
        self.initial_tokenizer = None

    def train(self, text, vocab_size, initial_tokenizer=None, verbose=False):
        init_time = time.time()
        self.extend_initial_tokenizer(initial_tokenizer)

        init_vocab_size = len(self.vocab)

        assert vocab_size >= init_vocab_size
        num_merges = vocab_size - init_vocab_size
        re_pattern = re.compile(self.pattern)
        text_chunks = re.findall(re_pattern, text)
        del text
        print("Time to split", time.time() - init_time)
        gc.collect()
        
        ids = self.init_encode(text_chunks)
        del text_chunks
        print("Time to encode", time.time() - init_time)
        gc.collect()
        
        self.merges, vocab_as_usize = rust_train_loop(ids, num_merges, init_vocab_size, self.vocab, self.merges)
        
        print("Time to complete", time.time() - init_time)
        # Convert vocab values from usize back to bytes
        self.vocab = {k: bytes(v) for k, v in vocab_as_usize.items()}

    def extend_initial_tokenizer(self, initial_tokenizer=None):
        if not initial_tokenizer:
            return None

        self.initial_tokenizer = initial_tokenizer

        tok_vocab = json.loads(initial_tokenizer._tokenizer.to_str())['model']['vocab']
        tok_merges = json.loads(initial_tokenizer._tokenizer.to_str())['model']['merges']

        self.vocab = {v: k.encode() for k, v in sorted(tok_vocab.items(), key=lambda x: x[1])}
        voc = initial_tokenizer.get_vocab()
        self.merges = [(voc[m.split()[0]], voc[m.split()[1]]) for m in tok_merges]

        return len(initial_tokenizer.get_vocab())

    def encode_chunk(self, chunk):
        return self.initial_tokenizer.encode(chunk, add_special_tokens=False)

    def init_encode(self, text_chunks):
        if not self.initial_tokenizer:
            return [list(ch.encode("utf-8")) for ch in text_chunks]
        else:
            with Pool(cpu_count()) as pool:
                return pool.map(self.encode_chunk, text_chunks)

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges:
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B", help="Tokenizer to extend")
        parser.add_argument("--text", type=str, default=False, help="corpus")
        parser.add_argument("--vocab_size", type=int, default=130000, help="vocab size")
        parser.add_argument('--use_fast', action='store_true', help="Use fast tokenizer")
        return parser.parse_args()

    args = parse_args()

    with open(args.text, 'r', encoding='utf-8') as f:
        text = f.read()

    init_tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=args.use_fast)
    
    tok = Tokenizer()
    tok.train(text, args.vocab_size, init_tok, verbose=True)

    with open('llama_3_ext.vocab', 'w') as f:
        json.dump(
            {tok.decode([k]):k for k in tok.vocab}, 
            f, 
            ensure_ascii=False, 
            indent=2
        )

    with open('llama_3_ext.merges', 'w') as f:
        f.writelines(
            [f"{tok.decode([k[0]])} {tok.decode([k[1]])}\n" for k in tok.merges]
        )

