import torch
from torch.utils import data
from PIL import Image
import os
import torchvision.transforms as transforms

class Dataset(data.Dataset):
  #'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, transform, root_dir):
        #'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.root_dir = root_dir
        self.transform = transform

  def __len__(self):
        #'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        #'Generates one sample of data
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        img_name = os.path.join(self.root_dir, ID)
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        y = self.labels[ID]

        return {'image' : image, 'label' : y}



