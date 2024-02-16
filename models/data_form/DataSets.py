from torch.utils.data import Dataset

class TabDictDataset(Dataset):
    def __init__(self, data):
        self.data_dict = data
        self.keys = list(data.keys())

    def __len__(self):
        key = self.keys[0]
        return len(self.data_dict[key])

    def __getitem__(self, index):
        batch = {k: self.data_dict[k][index] for k in self.keys}
        return batch
    