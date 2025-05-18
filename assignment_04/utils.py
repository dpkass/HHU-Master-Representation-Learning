import os
import re
from pathlib import Path
from functools import cache

import nltk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
MAX_SEQ_LEN = 256


class MovingAverage:
    def __init__(self, beta=0.99):
        self.beta = beta
        self.avg = None

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if self.avg is None:
            self.avg = value
        else:
            self.avg = self.beta * self.avg + (1 - self.beta) * value

    def get(self):
        return self.avg


class TextProcessor:

    def __init__(self):
        import nltk
        self.sent_tokenize = nltk.sent_tokenize
        nltk.download('punkt')

        self.CLEAN_HTML = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        self.CLEAN_PUNKT = re.compile('[' + re.escape('!"#$%&()*+,.:;<=>?@[\\]^_`{|}~') + ']')
        self.CLEAN_WHITE = re.compile(r'\s+')

    def clean_text(self, text):
        text = re.sub(self.CLEAN_HTML, ' ', text)
        text = re.sub(self.CLEAN_PUNKT, ' ', text.lower())
        text = re.sub(self.CLEAN_WHITE, ' ', text)
        return text.strip()

    def __call__(self, text):
        return [self.clean_text(sent) for sent in self.sent_tokenize(text)]


class IMDBDataset(torch.utils.data.Dataset):

    def __init__(self, train=True):
        super().__init__()
        split = "unsupervised" if train else "test"
        raw_dataset = load_dataset("imdb", split=split)
        self.tokens = tokenize_dataset(raw_dataset, f"imdb_{split}", "text")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]


@cache
def load_imdb_dataset(train=True):
    return IMDBDataset(train)


def add_special_tokens(tokens_a, token_b=None):
    tokens = torch.cat([
        torch.tensor([TOKENIZER.cls_token_id]),
        tokens_a,
        torch.tensor([TOKENIZER.sep_token_id])
    ])
    normal_mask = torch.tensor([False] + [True] * len(tokens_a) + [False])
    segment_id = torch.zeros_like(tokens)
    if token_b is not None:
        tokens = torch.cat([
            tokens,
            token_b,
            torch.tensor([TOKENIZER.sep_token_id])
        ])
        normal_mask = torch.cat([
            normal_mask,
            torch.tensor([True] * len(token_b) + [False])
        ])
        segment_id = torch.cat([
            segment_id,
            torch.ones(len(token_b) + 1, dtype=torch.int16)
        ])
    return dict(
        input_ids=tokens.long(),
        normal_mask=normal_mask,
        segment_ids=segment_id.long())


class SST2Dataset(torch.utils.data.Dataset):

    def __init__(self, train=True):
        super().__init__()
        split = "train" if train else "validation"
        self.raw_dataset = load_dataset("glue", "sst2", split=split)
        self.tokens = tokenize_dataset(self.raw_dataset, f"sst2_{split}", "sentence")

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        label = self.raw_dataset[index]['label']
        tokens = self.tokens[index][0]
        out = add_special_tokens(tokens)
        out.update(labels=label)
        return out


@cache
def load_sst2_dataset(train=True):
    return SST2Dataset(train)


def tokenize_dataset(dataset, name, key):
    path = Path('.cache')
    path.mkdir(exist_ok=True)
    path = path / f"{name}_tokens.pt"
    if path.exists():
        return torch.load(path)
    print(f"Tokenizing {name} dataset...")
    text_processor = TextProcessor()
    out = []
    for i in tqdm(range(len(dataset))):
        text = dataset[i][key]
        # If the item is a tuple, it is a labeled dataset
        if isinstance(text, tuple):
            text = text
        text = text_processor(text)
        ids = TOKENIZER(text,
                        max_length=MAX_SEQ_LEN,
                        add_special_tokens=False,
                        truncation=True)['input_ids']
        ids = [torch.tensor(ids, dtype=torch.int16, device='cpu') for ids in ids]
        out.append(ids)
    torch.save(out, path)
    return out


class SST2Model(torch.nn.Module):

    def __init__(self, bert_encoder, train_encoder=True):
        """
        Args:
            bert_encoder: An instance of a BERTEncoder
            train_encoder: wheter the encoder should be trained or not.
        """
        super().__init__()

        self.bert_encoder = bert_encoder
        for param in self.bert_encoder.parameters():
            param.requires_grad = train_encoder
        self.classifier = torch.nn.Linear(bert_encoder.d_model, 1, bias=False)

    def forward(self, input_ids):
        """
        Predicts the sentiment of a sentence (positive or negative)
        Args:
            input_ids: tensor of shape (batch_size, seq_len) containing the token ids of the sentences
        Returns:
            tensor of shape (batch_size) containing the predicted sentiment
        """
        h = self.bert_encoder(input_ids)
        return self.classifier(h[:, 0]).view(-1)
    

def train_sst2(bert_encoder, train_encoder=False, epochs=3, batch_size=256, lr=1e-3, device='cuda'):
    sst2_dataset = load_sst2_dataset(train=True)
    loader = DataLoader(sst2_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    sst2_model = SST2Model(bert_encoder, train_encoder=train_encoder).to(device).train()
    opt = torch.optim.AdamW(sst2_model.classifier.parameters(), lr=lr)
    loss_avg = MovingAverage()
    acc_avg = MovingAverage()
    for ep in range(epochs):
        with tqdm(loader, desc=f'Epoch {ep}') as pbar:
            for batch in pbar:
                opt.zero_grad(set_to_none=True)
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].float().to(device)
                logits = sst2_model(input_ids)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                opt.step()
                loss_avg.update(loss)
                acc_avg.update(((logits > 0) == labels).float().mean())
                pbar.set_postfix(
                    loss=loss_avg.get(),
                    acc=acc_avg.get()
                )
    return sst2_model


@torch.no_grad()
def validate_sst2(model, device):
    model.eval()
    dataset = load_sst2_dataset(train=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    accs = []
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        logits = model(input_ids)
        pred = logits > 0
        accs.append((pred == labels).float())
    return torch.cat(accs).mean().item() * 100                       

############################################################################################################


import torch
from math import prod

from typing import Callable, Dict, Optional, Tuple, Type, Union


# This function is new
def padded_stack(tensors, pad_length, dim=0, *, out=None):
    padded_tensors = []
    for tensor in tensors:
        padding = torch.zeros(pad_length - tensor.size(0), *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
        padded_tensor = torch.cat([tensor, padding], dim=0)
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors, dim=dim, out=out)


# This is copied from the PyTorch source code and only modified to allow padding
def padded_collate_tensor_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    max_length = max(t.size(0) for t in batch) # To know how much to pad
    out = None
    if elem.is_nested:
        raise RuntimeError(
            "Batches of nested tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )
    if elem.layout in {torch.sparse_coo, torch.sparse_csr, torch.sparse_bsr, torch.sparse_csc, torch.sparse_bsc}:
        raise RuntimeError(
            "Batches of sparse tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        # The shape/numel needed to be modifed here
        shape = len(batch), max_length, *elem.shape[1:]
        numel = prod(shape)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(*shape)
    return padded_stack(batch, max_length, dim=0, out=out)

