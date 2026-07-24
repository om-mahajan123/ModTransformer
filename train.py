import os
import math
import time
import csv

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
BATCH_SIZE = 128
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
train_data = PTBTrain(train_tokens, CONTEXT_WINDOW)
val_data = PTBTrain(val_tokens, CONTEXT_WINDOW)

training_batches = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_batches = DataLoader(
    dataset=val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)

model = Transformer(NUM_BLOCKS, D_MODEL, VOCAB_SIZE, NUM_HEADS, FFN_HIDDEN, CONTEXT_WINDOW)


def train_model(model, training_batches, val_batches, epochs, vocab_size,
                lr=0.001, device=None, checkpoint_path="data/weights/best_model.pt",
                log_path="data/logs/training_log.csv",
                batch_log_path="data/logs/batch_log.csv",
                progress_interval_sec=120):

    # Make sure output directories exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(batch_log_path), exist_ok=True)

    #Checking which device is available CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    #Setting up backprop loss function and parameter optimzing/gradient
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    train_loss = []
    val_loss = []
    best_val_loss = float("inf")
    start_epoch = 0

    #Resume from checkpoint if one exists
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}, loading...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    #Set up CSV logs
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "avg_train_loss", "avg_val_loss", "perplexity", "epoch_time_sec"])

    if not os.path.exists(batch_log_path):
        with open(batch_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "batch_idx", "train_loss"])

    total_batches = len(training_batches)
    print(f"Starting training | {total_batches} batches per epoch")

    try:
        for epoch in range(start_epoch, epochs):
            total_train_loss = 0
            model.train()
            epoch_start = time.time()
            last_progress_print = epoch_start

            with open(batch_log_path, "a", newline="") as batch_f:
                batch_writer = csv.writer(batch_f)

                for batch_idx, (batch_x, batch_y) in enumerate(training_batches):
                    #Forward pass
                    #Moving batch to GPU if available
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    logits = model(batch_x)
                    loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))

                    train_loss.append(loss.item())
                    total_train_loss += loss.item()
                    batch_writer.writerow([epoch, batch_idx, loss.item()])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #Tracking how long an epoch takes, long we have been training, many batches
                    now = time.time()
                    if now - last_progress_print >= progress_interval_sec:
                        elapsed = now - epoch_start
                        batches_done = batch_idx + 1
                        avg_time_per_batch = elapsed / batches_done
                        remaining_batches = total_batches - batches_done
                        eta_sec = remaining_batches * avg_time_per_batch

                        running_avg_loss = total_train_loss / batches_done
                        print(f"[Epoch {epoch}] Batch {batches_done}/{total_batches} "
                              f"| Avg Loss: {running_avg_loss:.4f} "
                              f"| Elapsed: {elapsed/60:.1f}m | ETA this epoch: {eta_sec/60:.1f}m")
                        last_progress_print = now

            #Validation set loss tracking
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
            epoch_time = time.time() - epoch_start
            perplexity = math.exp(avg_val_loss)

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss, perplexity, epoch_time])

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                }, checkpoint_path)
                print("Saved Weights (new best)")

            print(f"Epoch: {epoch} | Time: {epoch_time/60:.1f}m\n"
                  f"Training Loss | {avg_train_loss:.4f} | Validation Loss {avg_val_loss:.4f} | Perplexity: {perplexity:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current state before exiting...")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, checkpoint_path.replace(".pt", "_interrupted.pt"))
        print("Saved. Safe to exit.")

    return train_loss, val_loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint_path", type=str, default="data/weights/best_model.pt")
    parser.add_argument("--log_path", type=str, default="data/logs/training_log.csv")
    parser.add_argument("--batch_log_path", type=str, default="data/logs/batch_log.csv")
    parser.add_argument("--progress_interval_sec", type=int, default=120)
    args = parser.parse_args()

    train_model(
        model, training_batches, val_batches,
        epochs=args.epochs, vocab_size=VOCAB_SIZE,
        lr=args.lr,
        checkpoint_path=args.checkpoint_path,
        log_path=args.log_path,
        batch_log_path=args.batch_log_path,
        progress_interval_sec=args.progress_interval_sec,
    )