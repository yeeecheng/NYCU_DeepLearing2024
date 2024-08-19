import torchvision
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
from torch.utils.data import Dataset as torchData
import os
import json
# import torch.nn.functional as F
import torch

class LoadDataset(torchData):
    def __init__(self, data_path, json_path, objects_path, mode= "train", partial=1.0):
        super().__init__()

        self.data_path = data_path
        self.mode = mode
        self.partial = partial
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
        ])
        
        with open(json_path, 'r') as f:
            self.label_json_file = json.load(f)
            if self.mode == "train":
                self.imgs, self.labels = list(self.label_json_file.keys()), list(self.label_json_file.values())
            elif self.mode in ["test", "new_test"]:
                self.labels = self.label_json_file

        with open(objects_path, 'r') as f:
            self.obj_json_file = json.load(f)
        
        self.label_one_hot = torch.zeros(len(self.labels), len(self.obj_json_file))

        for i, label in enumerate(self.labels):
            for j in label:
                self.label_one_hot[i][self.obj_json_file[j]] = 1

    def __len__(self):
        return  int(len(self.labels) * self.partial) 

    def __getitem__(self, idx):

        if self.mode == "train":
            img_path = os.path.join(self.data_path, self.imgs[idx])
            return self.transform(imgloader(img_path)), self.label_one_hot[idx] 
        elif self.mode in ["test", "new_test"]:
            return self.label_one_hot[idx] 