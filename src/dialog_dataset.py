from torch.utils.data import Dataset
import torch
from .utils import get_tokenizer

class DialogDataset(Dataset):

    def __init__(self, dataset_path, tokenizer):

        self.tokenizer = tokenizer

        self.examples = [(self.tokenizer.encode(line.strip().split('\t')[0]).ids, self.tokenizer.encode(line.strip().split('\t')[1]).ids) for line in open(dataset_path, 'r', encoding='utf-8').readlines()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i][0]), torch.tensor(self.examples[i][1])
    

        

        
