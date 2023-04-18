from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, data, col_text, col_numeric, col_target):
        self.data = data
        self.target = data[col_target].values.reshape(-1, 1)
        self.col_text = col_text
        self.col_numeric = col_numeric

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):


        sample = {
            'encoding': numeric_data,
            'target': self.target[idx]
        }

        return sample
