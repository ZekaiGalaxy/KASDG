import pickle
import torch
import os
from torch.utils.data import Dataset

def filterout(x,y,ratio):
    x = x.split(' ')
    y = y.split(' ')
    common = 0
    for xx in x:
        if xx in y:
            common += 1
    if common >= ratio * len(x):
        return 1
    else:
        return 0

def save_model(model, save_path, step):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'model_state_dict':model.state_dict()}, save_path+f'/step_{step}.pt')
    
def write_list(lst,path):
    with open(path, 'wb+') as fp:
        pickle.dump(lst, fp)

def read_list(path):
    with open(path, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

class DatasetWrapper(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]