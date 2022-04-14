
import torch
import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, filepath_list, transform=None):
        datalist = []
        labellist = []
        for i in range(len(filepath_list)):
            datalist.extend(unpickle(filepath_list[i])[b"data"])
            labellist.extend(unpickle(filepath_list[i])[b"labels"])
        self.transform = transform
        self.label = labellist
        self.data = datalist

        self.datanum = 50000

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data_r = self.data[idx][0:1024]
        out_data_r = torch.from_numpy(out_data_r).float()
        out_data_g = self.data[idx][1024:2048]
        out_data_g = torch.from_numpy(out_data_g).float()
        out_data_b = self.data[idx][2048:3072]
        out_data_b = torch.from_numpy(out_data_b).float()

        out_data_r = out_data_r.reshape(32, 32)
        out_data_g = out_data_g.reshape(32, 32)
        out_data_b = out_data_b.reshape(32, 32)

        out_data = [out_data_r, out_data_g, out_data_b]
        out_data = np.stack([out_data_b, out_data_g, out_data_r])

        out_label = self.label[idx]
        self.out_label = torch.tensor(out_label)

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, self.out_label
