import torch
from torch.utils import data
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
from data_class_train import Dataset
from load import read_test, genTest
from densenet import ConvNet2, ConvNet3
import pandas as pd
import sys

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

def test(model, device, test_loader):
    #PATH = "./result_self/epoch:49_0.782626919940718.model"
    #PATH = "./result/epoch:18_0.7788017858015678.model"
    #model.load_state_dict(torch.load(PATH))
    model.eval()
    result = np.zeros((1, 14), dtype = np.float32)
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            result = np.concatenate((result, np.asarray(output)), axis = 0)
            # bs, ncrops, c, h, w = data.size()
            # result_ = model(data.view(-1, c, h, w)) # fuse batch size and ncrops
            # result_avg = result_.view(bs, ncrops, -1).mean(1) # avg over crops
            # result = np.concatenate((result, result_avg.cpu().numpy()), axis = 0)
            #return np.asarray(output)
            #pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
    result = np.delete(result, 0, axis = 0)
    print(result.shape)
    return result

def read_data(file_name):
    result = np.load(file_name)
    return result

def write_result(result, idx, file_name):
    idx = pd.DataFrame(idx, columns = ["Id"])
    result = pd.DataFrame(result, columns = ["Atelectasis","Cardiomegaly" ,"Effusion" ,"Infiltration" ,"Mass" ,"Nodule" ,"Pneumonia" ,"Pneumothorax" ,"Consolidation" ,"Edema" ,"Emphysema" ,"Fibrosis" ,"Pleural_Thickening" , "Hernia"])
    frames = [idx, result]
    output = pd.concat(frames, 1)
    output.to_csv(file_name, index = False)
    #pd.DataFrame(result, columns = ["id","Value"]).to_csv(outfile, index = False)
    return output

def avg_results(results, weights):
    sum_weight = 0.0
    new_r = []
    for i in range(len(results)):
        new_r.append(weights[i] * results[i]) 
        sum_weight += weights[i]
    print(sum_weight)
    new_r = np.asarray(new_r)
    print(new_r.shape)
    result = np.sum(new_r, axis = 0)/ sum_weight
    print(result.shape)
    return result

def read_data_1(file_name):
    result = np.load(file_name)
    return result



file_name = sys.argv[1]
outfile = sys.argv[2]
root_dir = sys.argv[3]
if file_name[-1] == '\r' or file_name[-1] == '\n':
    file_name = file_name[:-1]
if outfile[-1] == '\r' or outfile[-1] == '\n':
    outfile = outfile[:-1]
if root_dir[-1] == '\r' or root_dir[-1] == '\n':
    root_dir = root_dir[:-1]

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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(torch.tensor(mean), torch.tensor(std))])

transform_600 = transforms.Compose([
    transforms.Resize(685),
    transforms.CenterCrop(600),
    transforms.ToTensor(),
    transforms.Normalize(torch.tensor(mean), torch.tensor(std))])

# testing_set = Dataset(partition['train'], labels, transform)
# testing_generator = data.DataLoader(testing_set, **params)

testing_set_600 = Dataset(partition['train'], labels, transform_600, root_dir)
testing_generator_600 = data.DataLoader(testing_set_600, **params)

models = []
results = []

# epoch:14_0.8131208840273955.model 0.76795  drop 0.2?  0.77266
# epoch:10_0.8179188753750226.model  dropout 0.4   0.77434
# epoch:11_0.8223481139229248.model  0.5  jitter 0.2  0.76886
# #epoch:12_0.8278717317096778.model  0.5 jitter 0.3  
# epoch:12_0.8157414198121032.model  0.5 jitter 0.2  0.77238 #0.1-0.2 valid  
# epoch:16_0.8295748704517211.model  0.5 0.4 jitter 0.2  #0.1-0.2 vlaid
# epoch:24_0.7853215205018956.model  0.5 0.2 #0.2-0.3 valid

PATH_1 = "./result/epoch_10_0.8179188753750226.model" #dropout 0.4 0.2 0.2   0.77434
PATH_2 = "./result/epoch_14_0.8131208840273955.model" #dropout 0.2?          0.77266
PATH_3 = "./result/epoch_12_0.8157414198121032.model"   #drop 0.5 0.2         0.77238
PATH_4 = "./result/epoch_11_0.8223481139229248.model"   #drop 0.5 0.2        0.76886
PATH_5 = "./result/epoch_24_0.7853215205018956.model"  #drop 0.5 0.2       0.76221
PATH_6 = "./result/epoch_16_0.8295748704517211.model"  #drop 0.5 0.4      0.76997
PATH_7 = "./result/epoch-8_0.8163549656510083.model"  #no dropout           

if use_cuda:
    model_1 = ConvNet2(num_classes =14, dropout = 0.4).cuda()
    model_1.load_state_dict(torch.load(PATH_1))
    model_2 = ConvNet2(num_classes =14, dropout = 0.2).cuda()
    model_2.load_state_dict(torch.load(PATH_2))
    model_3 = ConvNet2(num_classes =14, dropout = 0.5).cuda()
    model_3.load_state_dict(torch.load(PATH_3))
    model_4 = ConvNet2(num_classes =14, dropout = 0.5).cuda()
    model_4.load_state_dict(torch.load(PATH_4))
    model_5 = ConvNet2(num_classes =14, dropout = 0.5).cuda()
    model_5.load_state_dict(torch.load(PATH_5))
    model_6 = ConvNet3(num_classes =14, dropout = 0.5).cuda()
    model_6.load_state_dict(torch.load(PATH_5))
    model_7 = ConvNet2(num_classes =14, dropout = 0).cuda()
    model_7.load_state_dict(torch.load(PATH_5))
else:
    model_1 = ConvNet2(num_classes =14, dropout = 0.4)
    model_1.load_state_dict(torch.load(PATH_1))
    model_2 = ConvNet2(num_classes =14, dropout = 0.2)
    model_2.load_state_dict(torch.load(PATH_2))
    model_3 = ConvNet2(num_classes =14, dropout = 0.5)
    model_3.load_state_dict(torch.load(PATH_3))
    model_4 = ConvNet2(num_classes =14, dropout = 0.5)
    model_4.load_state_dict(torch.load(PATH_4))
    model_5 = ConvNet2(num_classes =14, dropout = 0.5)
    model_5.load_state_dict(torch.load(PATH_5))
    model_6 = ConvNet3(num_classes =14, dropout = 0.5)
    model_6.load_state_dict(torch.load(PATH_5))
    model_7 = ConvNet2(num_classes =14, dropout = 0)
    model_7.load_state_dict(torch.load(PATH_5))

models.append(model_1)
models.append(model_2)
models.append(model_3)
models.append(model_4)
models.append(model_5)
models.append(model_6)
models.append(model_7)


###########if test#############
result_1 = test(model = models[0], device = device, test_loader = testing_generator_600)
np.save('result_1.npy', result_1)
result_2 = test(model = models[1], device = device, test_loader = testing_generator_600)
np.save('result_2.npy', result_2)
result_3 = test(model = models[2], device = device, test_loader = testing_generator_600)
np.save('result_3.npy', result_3)
result_4 = test(model = models[3], device = device, test_loader = testing_generator_600)
np.save('result_4.npy', result_4)
result_5 = test(model = models[4], device = device, test_loader = testing_generator_600)
np.save('result_5.npy', result_5)
result_6 = test(model = models[5], device = device, test_loader = testing_generator_600)
np.save('result_6.npy', result_6)
result_7 = test(model = models[6], device = device, test_loader = testing_generator_600)
np.save('result_7.npy', result_7)

results.append(result_1)
results.append(result_2)
results.append(result_3)
results.append(result_4)
results.append(result_5)
results.append(result_6)
results.append(result_7)


###########if load###################

# file_1 = 'result_1.npy'
# file_2 = 'result_2.npy'
# file_3 = 'result_3.npy'
# file_4 = 'result_4.npy'
# file_5 = 'result_5.npy'
# file_6 = 'result_6.npy'
# file_7 = 'result_7.npy'


# results.append(read_data(file_1))
# results.append(read_data(file_2))
# results.append(read_data(file_3))
# results.append(read_data(file_4))
# results.append(read_data(file_5))
# results.append(read_data(file_6))
# results.append(read_data(file_7))

weights = np.asarray([1.8, 1.8, 1.4, 1, 0, 1, 0.1])
res = avg_results(results, weights)

print("----testing completed----")

write_result(res, X, outfile)
