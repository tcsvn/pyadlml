import numpy as np
from pathlib import Path
import pandas as pd
from pyadlml.pipeline import Pipeline
from torch.utils.data import Dataset
import json
import pickle
import h5py

class TorchDataset(Dataset):
    def __init__(self, X, y, transforms: Pipeline, transform_params={}, class_encoding=True, use_cached=False, hdf5=False):
        import torch

        rand_str = self._get_rand_rep(transforms)
        sub_folder = 'train' if transforms.is_in_train_mode() else 'eval'

        #self.fp_tmp = Path(f'/mnt/ssd256/tmp/{rand_str}/{sub_folder}/')
        self.fp_tmp = Path(f"/home/chris/master_thesis/trainings_cache/{rand_str}/{sub_folder}/")
        self.hdf5 = hdf5

        if use_cached and self.fp_tmp.exists(): 
            with open(self.fp_tmp.joinpath('data.pt'), 'rb') as f:
                tmp_dict = pickle.load(f)

            self.classes_ = tmp_dict['classes']
            self.class_weights_ = tmp_dict['class_weights']
            self.nr_steps = tmp_dict['step']
            self.chunk_size = tmp_dict['chunk_size']
            self.shape = tmp_dict['shape']

            self.int2class_ = {k:v for k, v in enumerate(self.classes_)}
            self.int2class_func = np.vectorize(self.int2class_.get)
            self.class2int_ = {v:k for k, v in self.int2class_.items()}
            self.class2int_func = np.vectorize(self.class2int_.get)
        else:
            if transforms.is_in_train_mode():
                X_t, Y_t = transforms.fit_transform(X.copy(), y.copy(), **transform_params)
            else:
                X_t, Y_t = transforms.transform(X.copy(), y.copy(), **transform_params)

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
            import psutil
            self.shape = X_t.shape
            self.N = X_t.shape[0]
            mem_frac = 0.8
            if hdf5:
                self.fp_tmp.mkdir(exist_ok=True, parents=True)
                self.fp_tmp = Path(f'/mnt/ssd256/tmp/{rand_str}/{sub_folder}.hdf5')
                with h5py.File(self.fp_tmp, "w") as f:
                    f.create_dataset("X_t", shape=X_t.shape, dtype=np.float32)
                    f.create_dataset("y_t", shape=Y_t.shape, dtype=np.int64)
                    f.attrs['classes'] = self.classes_
                    f.attrs['shape'] = self.shape
                    f.attrs['class_weights'] = self.class_weights_

            if X_t.nbytes > psutil.virtual_memory().available*mem_frac or use_cached:
                self.fp_tmp.mkdir(exist_ok=True, parents=True)
                self.chunk_size = 100
                self.nr_steps = self.N//self.chunk_size
                for c in range(self.nr_steps+1):
                    torch.save(
                        torch.tensor(X_t[self.chunk_size*c:self.chunk_size*c+self.chunk_size]).to(torch.float32),
                        self.fp_tmp.joinpath(f'Xt_chunk_{c}.pt')
                    )
                    torch.save(
                        torch.tensor(Y_t[self.chunk_size*c:self.chunk_size*c+self.chunk_size]).long(), 
                        self.fp_tmp.joinpath(f'Yt_chunk_{c}.pt')
                    )

                # 
                tmp_dict = {}
                tmp_dict['classes'] = self.classes_
                tmp_dict['step'] = self.nr_steps
                tmp_dict['chunk_size'] = self.chunk_size
                tmp_dict['shape'] = self.shape
                tmp_dict['class_weights'] = self.class_weights_
                with open(self.fp_tmp.joinpath('data.pt'), 'wb') as f:
                    pickle.dump(tmp_dict, f)
            else:
                self.Xtr = torch.tensor(X_t).to(torch.float32)
                self.Ytr = torch.tensor(Y_t).long()
            

    def __len__(self):
        return self.shape[0]

    def _get_rand_rep(self, pipe):
        import hashlib
        return hashlib.md5(str(pipe).encode('utf-8')).hexdigest()
        #rand_str = "".join([str(s)for s in np.random.choice(10,10)])

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

        import torch
        if self.hdf5:
            print()
        if hasattr(self, 'chunk_size'):
            c = idx//self.chunk_size
            try:
                Xtr = torch.load(self.fp_tmp.joinpath(f'Xt_chunk_{c}.pt'))
                Ytr = torch.load(self.fp_tmp.joinpath(f'Yt_chunk_{c}.pt'))
            except FileNotFoundError:
                print()
            new_idx = idx - c*self.chunk_size
            return Xtr[new_idx], Ytr[new_idx]
        else:
            return self.Xtr[idx], self.Ytr[idx]
 