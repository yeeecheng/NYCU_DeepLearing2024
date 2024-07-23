import torch
import numpy as np
import os

class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        features = None
        for file in os.listdir(filePath):
            if features is None:
                features = np.load(os.path.join(filePath, file))
            else:
                features = np.concatenate((features, np.load(os.path.join(filePath, file))), axis= 0)
        features = np.expand_dims(features, axis= 1)
        return torch.tensor(features)

    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        labels = None
        for file in os.listdir(filePath):
            if labels is None:
                labels = np.load(os.path.join(filePath, file))
            else:
                labels = np.concatenate((labels, np.load(os.path.join(filePath, file))), axis= 0)
        return torch.tensor(labels)

    def __init__(self, mode):
        self.features = self._getFeatures(filePath= f'./dataset/{mode}/features/')
        self.labels = self._getLabels(filePath= f'./dataset/{mode}/labels/')

    def __len__(self):
        # implement the len method
        return len(self.labels)

    def __getitem__(self, idx):
        # implement the getitem method
        return self.features[idx].float(), self.labels[idx]