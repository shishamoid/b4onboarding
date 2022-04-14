import torchvision
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self,filepath_list, transform=None):
        #self.transform = transforms.Compose([transforms.ToTensor()])
        #for i in range(3):
        datalist=[]
        labellist = []
        for i in range(len(filepath_list)):
            #print(unpickle(filepath_list[i])[b"data"])
            datalist.extend(unpickle(filepath_list[i])[b"data"])
            labellist.extend(unpickle(filepath_list[i])[b"labels"])
        #data = unpickle(filepath)
        self.transform = transform
        #self.data = data[b"data"]
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
        #print(out_data.shape)

        out_label = self.label[idx]
        self.out_label = torch.tensor(out_label)
        #self.out_label = F.one_hot(self.out_label, num_classes=10)

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, self.out_label


#datasets = Mydatasets(bin)
"""
print(datasets.__len__())
print(datasets.__getitem__(2))
print(datasets.__getitem__(2)[0].shape)
print(datasets.__getitem__(2)[1].shape)
"""
