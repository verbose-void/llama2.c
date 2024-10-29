import xml.etree.ElementTree as ET
import torch
import os
import random


# get directory as current file path
DIR = os.path.dirname(os.path.realpath(__file__))


DATA_PATH = os.path.join(DIR, "..", "enwik8")
CACHE_PATH = os.path.join("enwik8_cache.pth")


def get_enwik9_text(path: str=DATA_PATH):
    entries = []

    with open(path, "r") as f:
        for line in f:
            entries.append(line.strip())
    
    return "\n".join(entries)


def get_train_eval_test_data():
    entries = get_enwik9_text()
    total_chars = len(entries)

    print(f"Total characters: {total_chars}")

    # let's do 90% train, 5% eval, 5% test (since it's not exactly 10M like the paper says)
    num_train = int(total_chars * 0.9)
    num_eval = int(total_chars * 0.05)
    num_test = total_chars - num_train - num_eval
    
    train_data = entries[:num_train]
    eval_data = entries[num_train:num_train + num_eval]
    test_data = entries[num_train + num_eval:]

    return train_data, eval_data, test_data


def _char_to_value(char: str):
    # return ord(char)
    # utf-8 encoded
    return ord(char)

# def _char_to_value(char: str):
#     # replace upper case letters with lower case letters
#     char = char.lower()

#     if len(char) == 2:
#         # for some reason i dot shows up like this. in that case, let's use the first char
#         # print(f"Invalid character: {char}")
#         # return ord(" ")
#         char = char[0]
#     assert len(char) == 1

#     char_ord = ord(char)
#     # ignore all characters that are not in the range of a-z
#     if char_ord < 97 or char_ord > 122:
#         return None
    
#     return char_ord


def get_tensor_enwik_data(subset: int = None, device: torch.device = torch.device("cpu"), use_cache: bool = True):
    # will cache the tensor data
    if os.path.exists(CACHE_PATH) and use_cache:
        print("Cache found, loading...")
        train_x, eval_x, test_x = torch.load(CACHE_PATH, map_location=device)
    else:
    
        print("Cache not found, generating...")
        train, eval, test = get_train_eval_test_data()

        train_chars = list(filter(lambda x: x is not None, map(_char_to_value, train)))
        eval_chars = list(filter(lambda x: x is not None, map(_char_to_value, eval)))
        test_chars = list(filter(lambda x: x is not None, map(_char_to_value, test)))

        print(f"Filtered total chars: train={len(train_chars)}, eval={len(eval_chars)}, test={len(test_chars)}, total={len(train_chars) + len(eval_chars) + len(test_chars)}")

        train_x = torch.tensor(train_chars, dtype=torch.float32, device=device)
        eval_x = torch.tensor(eval_chars, dtype=torch.float32, device=device)
        test_x = torch.tensor(test_chars, dtype=torch.float32, device=device)

        # replace the values with their unique value indices
        unique_chars = torch.unique(torch.cat([train_x, eval_x, test_x]))
        # use searchsorted
        train_x = torch.searchsorted(unique_chars, train_x)
        eval_x = torch.searchsorted(unique_chars, eval_x)
        test_x = torch.searchsorted(unique_chars, test_x)


        torch.save((train_x, eval_x, test_x), CACHE_PATH)
        print("Cache saved.")

    # count the number of unique characters in the total dataset
    unique_chars = torch.unique(torch.cat([train_x, eval_x, test_x]))
    print(f"Number of legal characters: {len(unique_chars)}")

    # get the global max and min for all datasets (for normalization)
    min_val, max_val = train_x.min(), train_x.max()
    min_val, max_val = min(min_val, eval_x.min()), max(max_val, eval_x.max())
    min_val, max_val = min(min_val, test_x.min()), max(max_val, test_x.max())
    # normalize for all datasets (such that min value is 0 and max value is 1)
    # train_x = (train_x - min_val) / (max_val - min_val)
    # eval_x = (eval_x - min_val) / (max_val - min_val)
    # test_x = (test_x - min_val) / (max_val - min_val)

    if subset is not None:
        train_x = train_x[:subset]
        eval_x = eval_x[:subset]
        test_x = test_x[:subset]

    return train_x, eval_x, test_x, unique_chars


class RandomSliceDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, context_len: int, randomized: bool = False, stride: int = None):
        assert len(data.shape) == 1

        self.randomized = randomized
        self.data = data
        self.context_len = context_len
        self.stride = stride if stride is not None else context_len

        # Calculate the number of valid starting indices
        self.valid_start_indices = (len(data) - context_len) // self.stride + 1

    def __len__(self):
        return self.valid_start_indices
    
    def __getitem__(self, idx):
        if not self.randomized:
            start_idx = idx * self.stride
        else:
            max_start = len(self.data) - self.context_len
            start_idx = random.randint(0, max_start)
            # Adjust start_idx to the nearest valid strided index
            start_idx = (start_idx // self.stride) * self.stride
        
        ctx_end_i = start_idx + self.context_len

        context = self.data[start_idx:ctx_end_i].long()
        target = self.data[start_idx + 1:ctx_end_i + 1].long()  # Right-shift target by one position
        
        assert len(context) == self.context_len, f"{len(context)} != {self.context_len}"
        assert len(target) == self.context_len, f"{len(target)} != {self.context_len}"
        return context, target.squeeze()

def get_dataloaders(batch_size: int, sequence_lengths: int, subset: int = None, stride: int = None, device: torch.device = torch.device("cpu"), use_cache: bool = True):
    train, eval, test, unique_chars = get_tensor_enwik_data(subset=subset, device=device, use_cache=use_cache)

    train_data = RandomSliceDataset(train, sequence_lengths, randomized=True, stride=stride)
    eval_data = RandomSliceDataset(eval, sequence_lengths, randomized=False, stride=stride)
    test_data = RandomSliceDataset(test, sequence_lengths, randomized=False, stride=stride)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, test_loader, unique_chars




class Task:

    def __init__(self, batch_size: int, device, vocab_size: int, sequence_lengths: int, num_workers: int = 0, **dataset_kwargs):
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs
        self.sequence_lengths = sequence_lengths


        self.train_loader, self.eval_loader, self.test_loader, self.unique_chars = get_dataloaders(
            batch_size=batch_size,
            sequence_lengths=sequence_lengths,
            **dataset_kwargs,
            device=device,
        )

        assert vocab_size == len(self.unique_chars), f"Vocab size mismatch: {vocab_size} != {len(self.unique_chars)}"

    def iter_batches(self, split: str):

        # get dl based on split
        if split == "train":
            dl = self.train_loader
        elif split == "val":
            dl = self.eval_loader
        elif split == "test":
            dl = self.test_loader
        else:
            raise ValueError(f"Invalid split: {split}")

        # do a forever loop (max steps are managed in the llama.c train code)
        while True:
            for x, y in dl:
                yield x, y.unsqueeze(-1)