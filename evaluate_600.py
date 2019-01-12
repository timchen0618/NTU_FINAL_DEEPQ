import torch
from torch.utils import data
from data_class_train import Dataset
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
from load import read_test, genTest
#from model_5_k24 import ConvNet2
from densenet import ConvNet2
import pandas as pd
import sys

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

def test(model, device, test_loader, PATH):

    model.load_state_dict(torch.load(PATH))
    model.eval()
    result = np.zeros((1, 14), dtype = np.float32)
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            result = np.concatenate((result, np.asarray(output)), axis = 0)

    result = np.delete(result, 0, axis = 0)
    print(result.shape)
    return result


def write_result(result, idx, file_name):
    idx = pd.DataFrame(idx, columns = ["Id"])
    result = pd.DataFrame(result, columns = ["Atelectasis","Cardiomegaly" ,"Effusion" ,"Infiltration" ,"Mass" ,"Nodule" ,"Pneumonia" ,"Pneumothorax" ,"Consolidation" ,"Edema" ,"Emphysema" ,"Fibrosis" ,"Pleural_Thickening" , "Hernia"])
    frames = [idx, result]
    output = pd.concat(frames, 1)
    output.to_csv(file_name, index = False)
    return output


file_name = sys.argv[1]
X = read_test(file_name)
partition = genTest(X)
params = {'batch_size': 32,
        'shuffle': False,
        'num_workers': 8}
print(len(partition))
labels = np.zeros(len(partition))

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(685),
    transforms.CenterCrop(600),
    transforms.ToTensor(),
    transforms.Normalize(torch.tensor(mean), torch.tensor(std))])

testing_set = Dataset(partition['train'], labels, transform, sys.argv[3])
testing_generator = data.DataLoader(testing_set, **params)

if use_cuda:
    convnet = ConvNet2(num_classes =14, dropout = 0.5).cuda()
else:
    convnet = ConvNet2(num_classes =14)
path = sys.argv[2]


result = test(
    model = convnet,
    device = device,
    test_loader = testing_generator,
    PATH = path
    )
print("----testing completed----")
outfile = sys.argv[4]
write_result(result, X, outfile)
