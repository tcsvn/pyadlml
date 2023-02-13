import numpy as np
import pandas as pd
from pyadlml.pipeline import Pipeline
from torch.utils.data import Dataset

class TorchDataset(Dataset):

    def __init__(self, X, y, transforms: Pipeline, transform_params={}, class_encoding=True):

        import torch
        X_t, Y_t = transforms.fit_transform(X.copy(), y.copy(), **transform_params)

       # Cast X_t and Y_t to numpy
        if isinstance(X_t, pd.DataFrame) or isinstance(X_t, pd.Series):
            X_t = X_t.values
        if isinstance(Y_t, pd.DataFrame) or isinstance(Y_t, pd.Series):
            Y_t = Y_t.values
        assert isinstance(X_t, np.ndarray) and isinstance(Y_t, np.ndarray)

        # Create Y_t class encoding
        self.classes_ = np.unique(Y_t.flatten())
 
        if class_encoding:
            self.int2class_ = {k:v for k, v in enumerate(self.classes_)}
            self.int2class_func = np.vectorize(self.int2class_.get)
            self.class2int_ = {v:k for k, v in self.int2class_.items()}
            self.class2int_func = np.vectorize(self.class2int_.get)
            Y_t = self.class2int_func(Y_t)

        # Compute class frequencies
        # Position 0 equal freq of class with idx zero
        count_per_class = np.bincount(Y_t.flatten())
        assert len(self.classes_) == len(count_per_class)
        self.class_weights_ = torch.tensor(
            count_per_class.sum()/(len(count_per_class)*count_per_class
        ),dtype=torch.float32)

        # Create torch tensor dataset and let torch infer datatype
        # What could go wrong
        self.Xtr = torch.tensor(X_t).to(torch.float32)
        self.Ytr = torch.tensor(Y_t).long()

    def __len__(self):
        return self.Xtr.shape[0]

    @property
    def class_weights(self):
        return self.class_weights_

    @property 
    def dct_class_weights(self):
        return {k:v for k, v in zip(self.class2int_,self.classes_)}


    @property
    def n_classes(self):
        return len(self.classes_)

    def int2class(self, i: int):
        return self.int2class_[i]

    def class2int(self, cls: str):
        return self.class2int_[cls]

    def __getitem__(self, idx):
        return self.Xtr[idx], self.Ytr[idx]
 