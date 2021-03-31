import torch
import random
from argparse import ArgumentParser
from utils import get_tokenizer
from model import *

parser = ArgumentParser(description='Test the Chit Chat system')

parser.add_argument('-decoding_strategy', type=str, default='top1', choices=['top1', 'topk', 'multinomial']) 

args = parser.parse_args()

def decode(logits, decoding_strategy='max', k=3, temp=0.4):
    tokenizer.decode(logits.topk(10)[1][0].numpy())
    if decoding_strategy=='top1':
        target = logits.max(1)[1]
    elif decoding_strategy=='topk':
        target = logits.topk(k)[1][0][random.randint(0, k-1)].unsqueeze(-1)
    else:
        target = torch.multinomial(logits.squeeze().div(temp).exp().cpu(), 1)
    return target

def evaluate(sentence):
    with torch.no_grad():
        target = torch.Tensor([tokenizer.token_to_id('<s>')]).long()
        output_sentence = []
        encoder_outputs, hidden = model.encoder(torch.Tensor(tokenizer.encode(sentence).ids).long().unsqueeze(-1))
        for t in range(MAX_LENGTH):
            # first input to the decoder is the <sos> token
            output, hidden = model.decoder(target, hidden, encoder_outputs)
            target = decode(output, decoding_strategy)
            if target.numpy() == tokenizer.token_to_id('</s>'):
                return tokenizer.decode(output_sentence)
            else:
                output_sentence.append(target.numpy()[0])
    return tokenizer.decode(output_sentence)

device = 'cpu'
#Load model 
#Tamainak egokitu zuen beharretara (Train fitxategiko berdinak izan behar dira)
INPUT_DIM = 10000
OUTPUT_DIM = 10000
ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
ENC_HID_DIM = 1024
DEC_HID_DIM = 1024
ATTN_DIM = 1024
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
#Create train iterator
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load('../model/model.pt', map_location=device))

tokenizer = get_tokenizer('../model/')
MAX_LENGTH = 30

#Print welcome message
print('-------------------------------')
print('Welcome to the Chit Chat system')
print("Write 'Bye' to end the system.")
print('-------------------------------')

#Main system loop
user = input('-')
model.eval()
decoding_strategy = args.decoding_strategy

while user != 'Bye':
    sentence = evaluate(user)
    print('-' + sentence.capitalize())
    user = input('-')
    
sentence = evaluate(user)
print('-' + sentence.capitalize())
