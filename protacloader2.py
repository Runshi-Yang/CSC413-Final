import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


def collater(data_list):
    batch = {}
    name = [x["name"] for x in data_list]
    ligase_pocket = [x["ligase_pocket"] for x in data_list]
    target_pocket = [x["target_pocket"] for x in data_list]
    PROTAC = [x["PROTAC"] for x in data_list]
    label = [x["label"] for x in data_list]

    batch["name"] = name
    batch["ligase_pocket"] = Batch.from_data_list(ligase_pocket)
    batch["target_pocket"] = Batch.from_data_list(target_pocket)
    batch["PROTAC"] = Batch.from_data_list(PROTAC)
    batch["label"]=torch.tensor(label)
    return batch


class PROTACSet(Dataset):
    def __init__(self, name, ligase_pocket, target_pocket, PROTAC, label):
        super().__init__()
        self.name = name
        self.ligase_pocket = ligase_pocket
        self.target_pocket = target_pocket
        self.PROTAC = PROTAC
        self.label = label


    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        sample = {
            "name": self.name[idx],
            "ligase_pocket": self.ligase_pocket[idx],
            "target_pocket": self.target_pocket[idx],
            "PROTAC": self.PROTAC[idx],
            "label": self.label[idx],
        }
        return sample


