import json
import copy
import os
import gc

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE

from utils import transformers_version_lower_than_445

def extend_tokenizer(
        initial_tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast, 
        vocab: str, 
        merges: str, 
        save_as: str
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    
    if isinstance(initial_tokenizer, str):
        init_tok = AutoTokenizer.from_pretrained(initial_tokenizer)
    else:
        init_tok = initial_tokenizer
    
    initial_tokenizer_json = json.loads(init_tok._tokenizer.to_str())

    if transformers_version_lower_than_445():
        with open(merges) as f:
            new_merges = f.readlines()
    
        if not isinstance(new_merges, list):
            merges = []
            for m in new_merges:
                if m.endswith("\n"):
                    merges.append(m[:-1])
                else:
                    merges.append(m)
            merges = [(merge.split()[0], merge.split()[1]) for merge in merges]
        else:
            merges = new_merges
        del new_merges
    else:
        with open(merges) as f:
            merges = [(merge[0], merge[1]) for merge in json.load(f)]
    
    with open(vocab) as f:
        vocab = {k:v for k,v in sorted(json.load(f).items(), key=lambda x: x[1])}
    
    vocab_fixed = copy.deepcopy(vocab)
    init_vocab_size = init_tok.vocab_size
    init_added_tokens_size = len(init_tok.get_added_vocab())
    if init_vocab_size < len(init_tok.get_vocab()):
        for k, v in vocab.items():
            if v >= init_vocab_size:
                vocab_fixed[k] = v + init_added_tokens_size

    del vocab
    gc.collect()
    
    initial_tokenizer_json['model']['merges'] = merges
    initial_tokenizer_json['model']['vocab'] = vocab_fixed

    print("Before change:", len(json.loads(init_tok._tokenizer.to_str())['model']['vocab']))
    print("After change:", len(initial_tokenizer_json['model']['vocab']))

    print("Before change:", len(json.loads(init_tok._tokenizer.to_str())['model']['merges']))
    print("After change:", len(initial_tokenizer_json['model']['merges']))

    tokenizer_json = init_tok.backend_tokenizer.to_str()
    tokenizer = Tokenizer.from_str(tokenizer_json)

    added_tokens = {token: idx for token, idx in init_tok.get_added_vocab().items()}
    combined_vocab = {**added_tokens, **vocab_fixed}

    bpe = BPE(combined_vocab, merges)

    new_tokenizer = Tokenizer(bpe)

    if tokenizer.pre_tokenizer:
        new_tokenizer.pre_tokenizer = tokenizer.pre_tokenizer

    if tokenizer.normalizer:
        new_tokenizer.normalizer = tokenizer.normalizer

    if tokenizer.decoder:
        new_tokenizer.decoder = tokenizer.decoder

    if tokenizer.post_processor:
        new_tokenizer.post_processor = tokenizer.post_processor

    extended_tokenizer = PreTrainedTokenizerFast(tokenizer_object=new_tokenizer)

    extended_tokenizer.save_pretrained(save_as)
    init_tok.save_pretrained("temp_tok")
    
    with open(f"{save_as}/tokenizer.json") as f:
        tok_json = json.load(f)

    tok_json['pre_tokenizer'] = initial_tokenizer_json['pre_tokenizer']
    tok_json['normalizer'] = initial_tokenizer_json['normalizer']
    tok_json['decoder'] = initial_tokenizer_json['decoder']
    tok_json['post_processor'] = initial_tokenizer_json['post_processor']

    with open(f"{save_as}/tokenizer.json", "w") as f:
        json.dump(tok_json, f, indent=2, ensure_ascii=False)

    with open("temp_tok/tokenizer_config.json") as f:
        tok_config_json = json.load(f)
    with open(f"{save_as}/tokenizer_config.json", "w") as f:
        json.dump(tok_config_json, f, indent=2, ensure_ascii=False)

    with open(f"temp_tok/special_tokens_map.json") as f:
        tok_config_json = json.load(f)
    with open(f"{save_as}/special_tokens_map.json", "w") as f:
        json.dump(tok_config_json, f, indent=2, ensure_ascii=False)

    os.system("rm -rf temp_tok")

    print("Check EOS token")
    print(f"EOS token of OLD tokenizer")
    print(f"Token {init_tok.eos_token} with id {init_tok.eos_token_id}")

    tt = AutoTokenizer.from_pretrained(save_as)
    print(f"EOS token of NEW tokenizer")
    print(f"Token {tt.eos_token} with id {tt.eos_token_id}")
    print(f"Test encode: {tt.encode(tt.eos_token)}")
    print(f"Test decode {tt.decode([tt.eos_token_id])}")

    return tt


if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--initial_tokenizer", type=str, default="meta-llama/Meta-Llama-3.1-8B-instruct", help="Tokenizer to extend")
        parser.add_argument("--vocab", type=str, default=False, help="vocab json file to add")
        parser.add_argument("--merges", type=str, default=False, help="merges txt/json file to add depending on Tokenizers version")
        parser.add_argument("--save_as", type=str, default=False, help="New tokenizer name / directory name to save")
        return parser.parse_args()

    args = parse_args()

    tt = extend_tokenizer(
            args.initial_tokenizer, 
            args.vocab, 
            args.merges, 
            args.save_as
    )