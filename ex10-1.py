import pandas as pd
import os
import string
import numpy as np
import glob
from torch.utils.data.dataset import Dataset

df = pd.read_csv("data/CH10/*.csv")

class TextGeneration(Dataset):
    def clean_test(self, txt):
        txt ="".join(v for v in txt if v not in string.punctuation).lower()
        return txt

