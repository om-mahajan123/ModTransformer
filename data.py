import os
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

VOCAB_SIZE = 3000

BATCH_SIZE = 32
CONTEXT_WINDOW = 128
D_MODEL = 256

NUM_HEADS = 4
FFN_HIDDEN = 512
NUM_BLOCKS = 3

#This is the tokenization cell
#TrainTokenizer uses HuggingFaces tokenizers library to create it's own token/vocab set because or data set is
#   rather small with only ~890,000 words
def TrainTokenizer(raw_data, save_name):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE, 
        special_tokens=["<unk>", "<s>", "</s>"]
    )

    temp_file = "temp_train_data.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(raw_data)
    
    tokenizer.train([temp_file], trainer)

    os.remove(temp_file)

    os.makedirs("data/tokenized", exist_ok=True)
    save_path = f"data/tokenized/{save_name}.json"
    tokenizer.save(save_path)

    print(f"Created tokenized dataset with {VOCAB_SIZE} tokens and saved to {save_path}")
    return tokenizer

#This uses the tokenizer we trained and actually tokenized the dataset so that now it's just a long list of
#   numbers/tokens 
def Tokenize(raw_data, save_name):
    my_tokenizer = TrainTokenizer(raw_data, save_name)
    tokenized_data = my_tokenizer.encode(raw_data).ids
    print("Tokenized all data\n")
    return tokenized_data

#This cell is where we create our training batches
#We use Pytorch dataset to act as a plucking tool to pluck out a single training/target
#   tensors from the dataset
class PTBTrain(Dataset):
    def __init__(self, token_list, context_window):
        self.token_list = token_list
        self.context_window = context_window
    
    def __len__(self):
        return len(self.token_list) - self.context_window

    def __getitem__(self, idx):
        x = torch.tensor(self.token_list[idx:idx + self.context_window], dtype=torch.long)
        y = torch.tensor(self.token_list[idx + 1:idx + self.context_window + 1], dtype=torch.long)

        return x, y