import os
import json
import math
import regex as re
import time
import gc


from enum import Enum
from functools import partial, reduce
from multiprocessing import Pool, cpu_count
from operator import iconcat
from typing import List, Callable, Dict

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from ext_llama import train_loop as rust_train_loop
from extend_hf_tokenizer import extend_tokenizer
from utils import transformers_version_lower_than_445

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LLAMA3_SPLIT_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

class StoppingStrategies(Enum):
    MAX_LENGTH = "max_length"
    FERTILITY_LIMIT = "fertility_limit"
    FERTILITY_PLATEAU = "fertility_plateau"


class Tokenizer:
    def __init__(self):
        self.merges: list = []
        self.pattern: str = LLAMA3_SPLIT_PATTERN
        self.special_tokens: dict = {}
        self.vocab: dict = self._build_vocab()
        self.initial_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None
        self.eval_fertility: list = []
        self.eval_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None
        self.stopping_strategy: StoppingStrategies.value = StoppingStrategies.MAX_LENGTH.value
        self.stoppage: bool = False
        self.fertility_limit: float = None
        self.stopping_sensitivity: float = None

    def train(
            self, 
            text: str, 
            eval_text: str, 
            eval_steps: float | int, 
            stopping_strategy: str, 
            fertility_limit: float, 
            stopping_sensitivity: float, 
            vocab_size: int, 
            initial_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None
    ) -> None:
        init_time: float = time.time()

        self.stopping_strategy = stopping_strategy
        self.fertility_limit = fertility_limit
        self.stopping_sensitivity = stopping_sensitivity

        self.extend_initial_tokenizer(initial_tokenizer)

        init_vocab_size: int = len(self.vocab)
        assert vocab_size >= init_vocab_size
        num_merges: int = vocab_size - init_vocab_size

        print("Splitting training corpus...")
        re_pattern = re.compile(self.pattern)
        text_chunks = re.findall(re_pattern, text)
        
        print("Splitting evaluation corpus if provided...")
        eval_text_chunks: list = eval_text.strip().split()
        eval_steps = int(eval_steps) if int(eval_steps) == eval_steps else math.ceil(eval_steps * num_merges)
    
        del text
        print("Time to split", time.time() - init_time)
        gc.collect()
        
        print("Encoding training corpus...")
        ids: list = self.init_encode(text_chunks)
        del text_chunks
        print("Time to encode", time.time() - init_time)
        gc.collect()
        
        # Getting fertility of initial tokenizer on the evaluation set
        self.create_and_eval_tok(eval_text_chunks, False)
        current_step: int = 0
        while current_step < num_merges or self.stoppage:
            self.merges, vocab_as_usize, ids, current_step = rust_train_loop(
                ids,
                num_merges=num_merges,
                init_vocab_size=init_vocab_size,
                vocab=self.vocab,
                merges=self.merges,
                resume_step=current_step,
                steps_per_call=eval_steps
            )
            self.vocab = {k: bytes(v) for k, v in vocab_as_usize.items()}
            self.create_and_eval_tok(eval_text_chunks)

        print("Time to complete", time.time() - init_time)


    def extend_initial_tokenizer(
            self, 
            initial_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None
    ) -> int:
        print("Initializing tokenizer...")
        if not initial_tokenizer:
            return None

        self.initial_tokenizer = initial_tokenizer

        tok_vocab: dict = json.loads(initial_tokenizer._tokenizer.to_str())['model']['vocab']
        tok_merges: list = json.loads(initial_tokenizer._tokenizer.to_str())['model']['merges']

        self.vocab = {v: k.encode() for k, v in sorted(tok_vocab.items(), key=lambda x: x[1])}
        voc: dict = initial_tokenizer.get_vocab()

        if transformers_version_lower_than_445():
            self.merges = [(voc[m.split(" ")[0]], voc[m.split(" ")[1]]) for m in tok_merges]
        else:
            self.merges = [(voc[m[0]], voc[m[1]]) for m in tok_merges]
        
        return len(initial_tokenizer.get_vocab())

    def encode_chunk(
            self, 
            chunk: str, 
            _eval: bool = False
        ) -> List[int]:
        if not _eval:
            tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.initial_tokenizer
        else:
            tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = self.eval_tokenizer
        
        return tokenizer.encode(chunk, add_special_tokens=False)

    def init_encode(
            self, 
            text_chunks: List[str], 
            _eval: bool = False
        ) -> List:
        encode_fn: Callable = (
            self.encode_chunk if not _eval else partial(self.encode_chunk, _eval=_eval)
        )

        if not self.initial_tokenizer:
            return [list(ch.encode("utf-8")) for ch in text_chunks]
        else:
            with Pool(cpu_count()) as pool:
                return pool.map(encode_fn, text_chunks)

    def _build_vocab(self) -> Dict:
        vocab: dict = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges:
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def decode(
            self, 
            ids: List[int]
        ) -> str:
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

    def create_and_eval_tok(
            self, 
            eval_text_chunks: List[str], 
            create: bool = True
        ) -> None:
        if not eval_text_chunks:
            print("No evaluation corpus provided. Proceeding w/o evaluation")
            return
        
        self.eval_tokenizer = self._create_eval_tok() if create else self.initial_tokenizer
        self._eval_tok(eval_text_chunks)

        os.system("rm -rf temp_eval_tok")

    def _create_eval_tok(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        # TODO can do this without writing to disc, but it's a minor optimization
        with open('temp.vocab', 'w') as f:
            json.dump(
                {self.decode([k]):k for k in self.vocab}, 
                f, 
                ensure_ascii=False, 
                indent=2
            )
        if transformers_version_lower_than_445():
            with open('temp.merges', 'w') as f:
                f.writelines(
                    [f"{tok.decode([k[0]])} {tok.decode([k[1]])}\n" for k in self.merges]
                )
        else:
            with open('temp.merges', 'w') as f:
                json.dump(
                    [(tok.decode([k[0]]), tok.decode([k[1]])) for k in self.merges],
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        return extend_tokenizer(
                self.initial_tokenizer, 
                'temp.vocab', 
                'temp.merges', 
                'temp_eval_tok'
        )

    def _eval_tok(
            self, 
            eval_text_chunks: List[str]
    ) -> None:       
        print("Encoding evaluation corpus...")
        init_eval_enc_time: float = time.time()
        eval_ids: list = self.init_encode(eval_text_chunks, _eval=True)
        print("Time to encode", time.time() - init_eval_enc_time)
        print("Evaluating tokenizer")
        flattened_ids: list = reduce(iconcat, eval_ids, [])
        eval_fertility: float = len(flattened_ids) / len(eval_text_chunks)
        print(f"No. Tokens: {len(flattened_ids)}")
        print(f"No. Words {len(eval_text_chunks)}")
        print(f"Fertility: {eval_fertility}")
        self.eval_fertility.append(eval_fertility)
        print(self.eval_fertility)

        if self.stopping_strategy == StoppingStrategies.FERTILITY_LIMIT and eval_fertility <= self.fertility_limit:
            self.stoppage = True
        
        # TODO this `-2` for the amount of previous steps to check is arbitrary, maybe make it a hyperparam?
        # TODO same goes for len(self.eval_fertility) > 2
        if (
            self.stopping_strategy == StoppingStrategies.FERTILITY_PLATEAU 
            and (
                self.eval_fertility[-2:] - eval_fertility <= self.stopping_sensitivity 
                and len(self.eval_fertility) > 2
            )
        ):
            self.stoppage = True
        
        del eval_ids
        del init_eval_enc_time
        del eval_fertility
        del flattened_ids
        gc.collect()

        
if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B", help="Tokenizer to extend")
        parser.add_argument("--text", type=str, default=False, help="corpus")
        parser.add_argument("--eval_text", type=str, required=False, default="", help="texts to evaluate on")
        parser.add_argument("--eval_steps", type=float, required=False, default=1.0, help="evaluate every N steps")
        parser.add_argument("--stopping_strategy", type=str, required=False, default=StoppingStrategies.MAX_LENGTH.value, help="Stopping strategy to follow. Defaults to MAX_LENGTH which means now stopping")
        parser.add_argument("--fertility_limit", type=float, required=False, default=1.55, help="Low bound of target fertility")
        parser.add_argument("--stopping_sensitivity", type=float, required=False, default=0.05, help="Difference over the last N (TBD) evals that hasn't been covered")
        parser.add_argument("--target_vocab_size", type=int, default=130000, help="vocab size")
        parser.add_argument('--use_fast', action='store_true', help="Use fast tokenizer")
        return parser.parse_args()

    args = parse_args()

    with open(args.text, 'r', encoding='utf-8') as f:
        text = f.read()

    if args.eval_text:
        with open(args.eval_text, 'r', encoding='utf-8') as f:
                eval_text = f.read()
    else:
        eval_text = ''

    init_tok: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=args.use_fast)
    
    tok: Tokenizer = Tokenizer()

    eval_steps: float = args.eval_steps
    assert ((int(eval_steps) == eval_steps and eval_steps >= 1) or 0 <= eval_steps <= 1), "eval_steps needs to either be a float between 0 and 1 or an integer larger or equal to 1" 

    tok.train(text, eval_text, args.eval_steps, args.stopping_strategy, args.fertility_limit, args.stopping_sensitivity, args.target_vocab_size, init_tok)

    with open('llama_3_ext.vocab', 'w') as f:
        json.dump(
            {tok.decode([k]):k for k in tok.vocab}, 
            f, 
            ensure_ascii=False, 
            indent=2
        )

    if transformers_version_lower_than_445():
        with open('llama_3_ext.merges', 'w') as f:
            f.writelines(
                [f"{tok.decode([k[0]])} {tok.decode([k[1]])}\n" for k in tok.merges]
            )
    else:
        with open('llama_3_ext.merges', 'w') as f:
            json.dump(
                [(tok.decode([k[0]]), tok.decode([k[1]])) for k in tok.merges],
                f,
                ensure_ascii=False,
                indent=2
            )

    if tok.eval_fertility:
        save_as: str = "fertility_plot.png"
        print(f"Saving fertility plot as {save_as}")
        fig, ax = plt.subplots()
        ax.plot(range(len(tok.eval_fertility)), tok.eval_fertility, marker='o', label='Fertility')
        ax.set_xlabel('Evaluation Step')
        ax.set_ylabel('Fertility')
        ax.set_title('Tokenizer Fertility Over Time')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_as)
