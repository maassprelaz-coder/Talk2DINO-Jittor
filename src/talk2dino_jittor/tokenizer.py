"""Official OpenAI BPE tokenizer with a framework-neutral NumPy interface."""
from functools import lru_cache
import numpy as np
from ._vendor.simple_tokenizer import SimpleTokenizer

@lru_cache()
def _get_tokenizer(): return SimpleTokenizer()

def tokenize(texts,context_length=77,truncate=False):
    if isinstance(texts,str): texts=[texts]
    tokenizer=_get_tokenizer()
    start=tokenizer.encoder["<|startoftext|>"]
    end=tokenizer.encoder["<|endoftext|>"]
    output=np.zeros((len(texts),context_length),dtype=np.int32)
    for row,text in enumerate(texts):
        tokens=[start]+tokenizer.encode(text)+[end]
        if len(tokens)>context_length:
            if not truncate: raise ValueError("Text exceeds CLIP context length")
            tokens=tokens[:context_length]; tokens[-1]=end
        output[row,:len(tokens)]=tokens
    return output
