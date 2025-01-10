# TikToken-like tokenizer extension
TikToken-like tokenizer extension implementation

Make sure you have the correct python dependencies (maturin is one of them)

```console
maturin develop
```

### Create extended vocab, merge files

```console
python3 llama_ext.py --text test.txt --vocab_size 128050
```

#### Arguments

- `tokenizer`: Tokenizer to extend
- `text`: Training corpus
- `eval_text`: Texts to evaluate on
- `eval_steps`: Evaluate every N steps, defaults to 1.0 which means evaluate once at the end. A float input represents a fraction of the max_length, while an int input denotes the exact number of steps between each evaluation round.
- `stopping_strategy`: Stopping strategy to follow. Defaults to `max_length` which means no stopping. Choices: `max_length`, `fertility_limit`, `fertility_plateau`
- `fertility_limit`: Low bound of target fertility, used with `fertility_limit` stopping strategy
- `stopping_sensitivity`: Difference over the last N (TBD) eval steps that hasn't been reached
- `target_vocab_size`: Maximum vocab size to reach
- `use_fast`: Some tokenizers need to be initialized as a fast tokenizer

### Extend existing tokenizer

#### Simple example
```console
python3 extend_hf_tokenizer.py --initial_tokenizer meta-llama/Meta-Llama-3.1-8B-instruct --vocab llama_3_ext.vocab --merges llama_3_ext.merges --save_as krikri_tokenizer
```

#### Arguments

- `initial_tokenizer`: Tokenizer that we extended
- `vocab`: Output vocab from previous script
- `merges`: Output merges from previous script
- `save_as`: Directory name of new tokenizer
