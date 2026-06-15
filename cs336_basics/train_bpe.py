import re
import regex
from collections import Counter
import sys

#Given a tuple of tokens, and a pair to merge, return
#tuple of merged tokens
def merge(tpl: tuple, pair: tuple) -> tuple:
   b1, b2 = pair[0], pair[1]
   ret_val = ()
   i = 0
   while i <=len(tpl) - 1:
      x = tpl[i]
      if i < len(tpl) - 1:
         y = tpl[i+1]
         if x == b1 and y == b2:
            ret_val += (b1 + b2, )
            i += 2
         else:
            ret_val += (x, )
            i += 1
      else:
         ret_val += (x, )
         i += 1

   return ret_val

def train_bpe(input_path: str, 
              vocab_size: int, 
              special_tokens: list[str]):
   f = open(input_path)
   text = f.read()
   pattern = "|".join([re.escape(x) for x in special_tokens])
   parts = re.split(pattern, text)

   PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

   token_counts: tuple[bytes] = Counter()
   vocab = {x: bytes([x]) for x in range(256)}
   vocab[256] = "<|endoftext|>".encode()

   # First create token counts
   for part in parts:
      words = regex.findall(PAT, part)
      for word in words:
         tups = tuple(bytes([b]) for b in word.encode("utf-8"))
         token_counts[tups] += 1

   merges: list[tuple[bytes, bytes]] = []
   while len(vocab) < vocab_size:
      # Every iteration, create pair count using token counts
      pair_counts:Counter[bytes, bytes] = Counter()
      for tpl, cnt in token_counts.items():
         for x, y in zip(tpl[0: len(tpl) - 1], tpl[1: len(tpl)]):
            pair_counts[(x, y)] += cnt


      # Find the token pair to merge
      best_pair = max(pair_counts, key=lambda k: (pair_counts[k], k))
      b1, b2 = best_pair[0], best_pair[1]

      #Now create a new token count
      token_counts1: tuple[bytes] = Counter()
      for tpl, cnt in token_counts.items():
         token_counts1[merge(tpl, best_pair)] = cnt

      token_counts = token_counts1
      vocab[len(vocab)] = b1 + b2 
      merges.append((b1, b2))

   return vocab, merges

if __name__ == "__main__":
   vocab, merges = train_bpe('sample.txt', 280, ['<|endoftext|>'])
