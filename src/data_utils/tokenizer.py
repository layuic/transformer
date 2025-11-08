import torch

class SimpleTokenizer:
    def __init__(self, texts, level="char", min_freq=1, specials=("<pad>", "<unk>", "<s>", "</s>")):
        self.level = level
        self.pad, self.unk, self.bos, self.eos = specials
        from collections import Counter
        tokens = []
        for t in texts:
            toks = list(t) if level == "char" else t.split()
            tokens.extend(toks)
        cnt = Counter(tokens)
        vocab = [self.pad, self.unk, self.bos, self.eos]
        vocab += [t for t, f in cnt.items() if f >= min_freq and t not in vocab]
        self.stoi = {s: i for i, s in enumerate(vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
    def encode(self, text, max_len):
        toks = list(text) if self.level == "char" else text.split()
        ids = [self.stoi.get(t, self.stoi[self.unk]) for t in toks][:max_len]
        ids = ids + [self.stoi[self.pad]] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)
    def vocab_size(self):
        return len(self.stoi)




