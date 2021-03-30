import torch

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

def train_tokenizer(input_path, output_path, vocab_size=10000):

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[input_path], vocab_size=vocab_size, special_tokens=["[PAD]", "<s>", "</s>", "<unk>"])
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.save_model(output_path)
    return tokenizer

def get_tokenizer(path):
    tokenizer = ByteLevelBPETokenizer(path + 'vocab.json', path + 'merges.txt')
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    return tokenizer

