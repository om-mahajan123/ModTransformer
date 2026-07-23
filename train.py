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

from model import Transformer
from data import PTBTrain
from data import Tokenize

VOCAB_SIZE = 3000
BATCH_SIZE = 32
CONTEXT_WINDOW = 128
D_MODEL = 256
NUM_HEADS = 4
FFN_HIDDEN = 512
NUM_BLOCKS = 3


# 1. Pull the raw data from our files
train_raw = open('data/ptb.train.txt', 'r').read()
test_raw = open('data/ptb.test.txt', 'r').read()
val_raw = open('data/ptb.val.txt', 'r').read()

# 2. Tokenize the train/validation data at once
train_tokens = Tokenize(train_raw, "ptb.train_tokens.txt")
val_tokens = Tokenize(val_raw, "ptb.val_token.txt")

# 3. Setting up Data pulling and batching
#   ptb_dataset pulls a single sample of tokens from our tokenized data
train_data = PTBTrain(train_tokens, CONTEXT_WINDOW)
val_data = PTBTrain(val_tokens, CONTEXT_WINDOW)

#Actually sets up the training/label batches which we feed into Transformer and eventually calculate loss on
training_batches = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

#Setting up the validation batches to ensure the model isn't just overfitting and memorizing our training data
val_batches = DataLoader(
    dataset=val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)

# 4. Training the Transformer
model = Transformer(NUM_BLOCKS, D_MODEL, VOCAB_SIZE, NUM_HEADS, FFN_HIDDEN, CONTEXT_WINDOW)
def train_model(model, training_batches, val_batches, epochs, vocab_size, 
                lr=0.001, device=None, checkpoint_path="data/weights/best_model.pt"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    train_loss = []
    val_loss = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        total_train_loss = 0
        f_pass = 0
        model.train()
        for batch_x, batch_y in training_batches:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            #Forward Pass through entire transformer
            #Shape: (batch_size, context_window, vocab_size)
            logits = model(batch_x)

            #Takes in Shape: (batch_size * context_window, d_model) |   (batch_size * context_window)
            #We don't care if the prediction came from batch 2 or 30 as long as we keep the context_window in order
            loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
            train_loss.append(loss.item())
            total_train_loss += loss.item()

            f_pass += 1
            if f_pass % 300 == 0:
                print(f"Current Loss: {total_train_loss/f_pass}")

            #Set up optimizer to update parameters based on gradients
            optimizer.zero_grad()   #Removes the gradients and sets it to 0, otherwise we would accumulate gradients
            loss.backward()
            optimizer.step()

        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for val_x, val_y in val_batches:
                val_x, val_y = val_x.to(device), val_y.to(device)
                logits = model(val_x)

                loss = criterion(logits.view(-1, vocab_size), val_y.view(-1))
                val_loss.append(loss.item())
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(training_batches)
        avg_val_loss = total_val_loss / len(val_batches)

        #Saving the weights where validation loss is the best
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), checkpoint_path)
            best_val_loss = avg_val_loss
            print("Saved Weights")

        #Standard measurement for Language Models
        #Measures how confident a model is in it's predicition
        perplexity = math.exp(avg_val_loss)
        print(f"Epoch: {epoch}\nTraining Loss | {avg_train_loss} Validation Loss {avg_val_loss} | Perplexity: {perplexity}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint_path", type=str, default="data/weights/best_model.pt")
    args = parser.parse_args()

    train_model(
        model, training_batches, val_batches,
        epochs=args.epochs, vocab_size=VOCAB_SIZE,
        lr=args.lr, checkpoint_path=args.checkpoint_path
    )