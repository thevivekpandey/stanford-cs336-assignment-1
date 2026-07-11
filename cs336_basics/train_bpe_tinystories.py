from train_bpe import train_bpe
import pickle

if __name__ == "__main__":
   input_path = "TinyStories-train.txt"
   vocab, merges = train_bpe(input_path, 10000, ["<|endoftext|>"])
      
   with open("vocab.pkl", "wb") as f:
      pickle.dump(vocab, f)

   with open("merges.pkl", "wb") as f:
      pickle.dump(merges, f)
      
