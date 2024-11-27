# LlamaTiktokenExt
Llama tokenizer extension implementation

### Clone repo
```console
git clone git@github.com:LeonVouk/ScuffedExtLlama.git
cd ScuffedExtLlama
```

Make sure you have the correct python dependencies (maturin is one of them)

```console
maturin develop
```

### Create extended vocab, merge files

```console
python3 ext_llama.py --text test.txt --vocab_size 128050
```

### Extend existing tokenizer

```console
python3 extend_hf_tokenizer.py --initial_tokenizer meta-llama/Meta-Llama-3.1-8B-instruct --vocab llama_3_ext.vocab --merges llama_3_ext.merges --save_as /krikri_tokenizer
```
