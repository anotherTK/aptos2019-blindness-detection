
import os
import torch
import torch.utils.data
import json
from PIL import Image

class BlindnessDataset(torch.utils.data.Dataset):

    def __init__(self, root, stage, transforms=None):
        super(BlindnessDataset, self).__init__()

        self.root = root
        self.stage = stage
        self.transforms = transforms

        assert self.stage in ['train', 'val', 'test']

        self.data = json.load(open(os.path.join(self.root, self.stage + '.json')))

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = self.data[index]

        img = Image.open(os.path.join(self.root, data[0] + '.jpg'))
        if self.transforms:
            img = self.transforms(img)

        if self.stage == 'test':
            return img, data[0]

        return img, torch.tensor(data[1])

                