from collections.abc import Iterable, Iterator
import re
import regex
import pickle

MAX = 100000000
class Tokenizer:
   def __init__(self, 
                vocab: dict[int, bytes], 
                merges: list[tuple[bytes, bytes]], 
                special_tokens: list[str] | None =None):
      self.token_to_id: dict[bytes, int] = {}
      for idx, token in vocab.items():
         self.token_to_id[token] = idx
      self.vocab = vocab
      self.merge_set = set()
      self.merge_to_idx: dict[(bytes, bytes), int] = {}
      i = 0
      for merge in merges:
         self.merge_set.add(merge)
         self.merge_to_idx[merge] = i
         i += 1
      self.special_tokens = special_tokens

   @classmethod
   def from_files(cls, 
                  vocab_filepath: str, 
                  merges_filepath: str, 
                  special_tokens: list[str] | None =None):

      with open(vocab_filepath, 'rb') as f:
         vocab = pickle.load(f)
 
      with open(merges_filepath, 'rb') as f:
         merges = pickle.load(f)

      return cls(vocab, merges, special_tokens)

   def plain_tokens_to_idx(self, tokens):
      x = [self.token_to_id[t] for t in tokens]
      return x

   def find_index_to_merge(self, tups):
      tup_len = len(tups)
      min_idx = MAX
      start = -1
      for i in range(tup_len - 1):
         x = tups[i]
         y = tups[i + 1]
         if (x, y) in self.merge_to_idx:
            if self.merge_to_idx[(x, y)] < min_idx:
               min_idx = self.merge_to_idx[(x, y)]
               start = i
      #print(f"Finding index: {tups} => {start}")
      return start

   def merge_tokens(self, tups, idx):
      i = 0
      output = ()
      while i <= len(tups) - 1:
         if i == idx and i < len(tups) - 1:
            output += (tups[i]  +  tups[i + 1], )
            i += 2
         else:
            output += (tups[i], )
            i += 1
            
      #print(f"Merging: input = {tups}, output = {output}")
      return output

   def encode(self, text: str) -> list[int]:
      if self.special_tokens:
         pattern = "(" + "|".join([re.escape(x) for x in self.special_tokens])  + ")"
         parts = re.split(pattern, text)
      else:
         parts = [text]

      PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

      encoded: list[int] = []

      for part in parts:
         if self.special_tokens and part in self.special_tokens:
            encoded.append(self.token_to_id[part.encode("utf-8")])
         else:
            words = regex.findall(PAT, part)
            for word in words:
               tups = tuple(bytes([b]) for b in word.encode("utf-8"))
               while True:
                  idx = self.find_index_to_merge(tups)
                  if idx == -1:
                     encoded.extend(self.plain_tokens_to_idx(tups))
                     break
                  else:
                     tups = self.merge_tokens(tups, idx)
           
      return encoded

   def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
      for line in iterable:
         yield from self.encode(line)

   def decode(self, ids:list[int]) -> str:
      a = [self.vocab[i] for i in ids]
      x = b"".join(a)
      y = x.decode("utf-8", errors='replace')
 
      return y


if __name__ == "__main__":
   tokenizer = Tokenizer.from_files("vocab.pkl", "merges.pkl", ["<|endoftext|>"])
   f = open('sample.txt')
   for tokens in tokenizer.encode_iterable(f):
      print(tokens)
