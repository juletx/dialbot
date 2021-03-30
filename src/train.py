import torch
from model import Seq2Seq, Encoder, Decoder, Attention
from torch.utils.tensorboard import SummaryWriter
import math
import time
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import train_tokenizer
from dialog_dataset import DialogDataset
from torch.utils.data import RandomSampler, DataLoader

def generate_batch(data_batch):
    src_batch, trg_batch = [], []
    for example in data_batch:
        src_batch.append(example[0][:MAX_LENGTH])
        trg_batch.append(example[1][:MAX_LENGTH])
    return nn.utils.rnn.pad_sequence(src_batch, tokenizer.token_to_id('[PAD]')), nn.utils.rnn.pad_sequence(trg_batch, tokenizer.token_to_id('[PAD]'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
MAX_LENGTH = 50

#Tamainak egokitu zuen beharretara
INPUT_DIM = 10000
OUTPUT_DIM = 10000
ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
ENC_HID_DIM = 1024
DEC_HID_DIM = 1024
ATTN_DIM = 1024
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2

#Train tokenizer
tokenizer = train_tokenizer('../data/train.txt', '../model/', INPUT_DIM)

dataset = DialogDataset('../data/train.txt', tokenizer)

#Create train iterator
train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE, collate_fn=generate_batch)

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

#Ignore the index of the padding
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'))

def train(model: nn.Module,
          train_dataloader: DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0
    
    for iteration, (src, trg) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):

        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        if iteration % 1000 == 0 and iteration > 0:
            print(f'\tTrain Loss: {epoch_loss/iteration:.3f} | Train PPL: {math.exp(epoch_loss/iteration):7.3f}')
            tb_writer.add_scalar('loss', epoch_loss/iteration, iteration/1000)
            tb_writer.add_scalar('train_ppl', math.exp(epoch_loss/iteration) , iteration/1000)
            # Save checkpoint
            torch.save(model.state_dict(), '../model/model.pt')


        epoch_loss += loss.item()

    return epoch_loss / len(train_dataloader)

def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 40
CLIP = 5.0

best_valid_loss = float('inf')

tb_writer = SummaryWriter('../logs/')

for epoch in tqdm(range(N_EPOCHS)):

    start_time = time.time()

    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')

