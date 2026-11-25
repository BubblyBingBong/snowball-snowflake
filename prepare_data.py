"""Download MNIST and CIFAR-10 into data/ in the format the experiments expect.

MNIST:    torchvision download into data/MNIST.
CIFAR-10: the Hugging Face copy (uoft-cs/cifar10, parquet), decoded to data/cifar10.pt =
          {"train": (X uint8 [50000,32,32,3], y int64), "test": (X uint8 [10000,32,32,3], y int64)}.
          The 5,000-image validation split is a fixed random subset of this training order, so use this source
          (torchvision's CIFAR-10 has a different training order and would give a different split).

Usage: python prepare_data.py
"""
import io, os, urllib.request
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import datasets

HF = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"
os.makedirs("data/cifar10_hf", exist_ok=True)

datasets.MNIST("data", train=True, download=True)
datasets.MNIST("data", train=False, download=True)

out = {}
for split in ["train", "test"]:
    path = f"data/cifar10_hf/{split}.parquet"
    if not os.path.exists(path):
        print(f"downloading CIFAR-10 {split} ...", flush=True)
        urllib.request.urlretrieve(HF.format(split), path)
    t = pq.read_table(path).to_pydict()
    X = np.stack([np.array(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["img"]])
    out[split] = (torch.from_numpy(X), torch.tensor(t["label"], dtype=torch.int64))
    print(f"CIFAR-10 {split}: {tuple(X.shape)}", flush=True)
torch.save(out, "data/cifar10.pt")
print("saved data/cifar10.pt")
