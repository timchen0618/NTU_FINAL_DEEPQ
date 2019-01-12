import torch
from torch.utils import data
from data_classes import Dataset
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
from load import read_data, genLabels_Partition, read_data_self_
#from model_5_k24_2 import ConvNet2
from densenet import ConvNet2, ConvNet3
from torchsummary import summary
from sklearn.metrics import roc_auc_score
from densenet import densenet121
import sys

import os
#import tensorflow as tf

#####################################
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.20
#set_session(tf.Session(config=config))

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True



def batch2Traindata(local_batch, local_labels):
    local_batch = 0
    return local_batch

def test_valid(model, device, valid_loader):
    model.eval()
    result = np.zeros((1, 14), dtype = np.float32)
    labels = np.zeros((1, 14), dtype = np.float32)
    with torch.no_grad():
        for local_batch in valid_loader:
            data, label_batch = local_batch['image'].to(device), local_batch['label'].to(device)
            output = model(data)
            result = np.concatenate((result, np.asarray(output)), axis = 0)
            labels = np.concatenate((labels, label_batch.cpu().numpy()), axis = 0)

    result = np.delete(result, 0, axis = 0)
    labels = np.delete(labels, 0, axis = 0)
    print(result.shape)
    print(labels.shape)
    score = roc_auc_score(labels, result)
    print(score)
    return score

def test_train(model, device, train_loader):
    model.eval()
    result = np.zeros((1, 14), dtype = np.float32)
    labels = np.zeros((1, 14), dtype = np.float32)
    with torch.no_grad():
        print(device)
        for local_batch in train_loader:
            data, label_batch = local_batch['image'].to(device), local_batch['label'].to(device)
            output = model(data)
            result = np.concatenate((result, np.asarray(output)), axis = 0)

    result = np.delete(result, 0, axis = 0)
    labels = np.delete(labels, 0, axis = 0)
    print(result.shape)
    print(labels.shape)
    score = roc_auc_score(labels, result)
    print(score)
    return score

def train(convnet, optimizer, training_batch, labels):
    loss = 0
    optimizer.zero_grad()
    training_batch, labels = training_batch.to(device), labels.to(device)
    output = convnet(training_batch)
    loss_function = nn.BCELoss(reduce = False)
    loss = loss_function(output, target)
    loss.backward()

    return loss


def trainIters(partition, labels, params, max_epochs, convnet, transform):

    summary(convnet, (3, 800,800))

    loss_function = nn.BCELoss()
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    valid_transform = transforms.Compose([
            transforms.Resize(685),
            transforms.CenterCrop(600),
            transforms.ToTensor(),
            transforms.Normalize(torch.tensor(mean), torch.tensor(std))
        ])

    training_set = Dataset(partition['train'], labels, transform)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['validation'], labels, valid_transform)
    validation_generator = data.DataLoader(validation_set, **params)


    # Loop over epochs
    loss_history = []
    score_history = []
    plat_count = 0
    for epoch in range(max_epochs):
        losses = 0
        lr = 1e-5
        if epoch % 5 == 0:
            lr = lr/ 1.2 
        optimizer = optim.Adam(convnet.parameters(), lr = lr)

        for local_batch in training_generator:
            
            optimizer.zero_grad()
            training_batch, labels = local_batch['image'].to(device), local_batch['label'].to(device)
            output = convnet(training_batch)
            loss = loss_function(output, labels)
            losses += loss.item()
            loss.backward()
            optimizer.step()
            
        if len(loss_history) != 0:
            if loss_history[len(loss_history) - 1] - losses <= 0.01:
                plat_count += 1
        loss_history.append(losses)
        print("epoch: " + str(epoch) + "  loss_history = ")
        print(loss_history)
        
        # Validation
        score = test_valid(convnet, device, validation_generator)
        score_history.append(score)
        print(score_history)
        path = "./result/epoch:" + str(epoch) + "_" + str(score) + ".model"
        if score >= max(score_history) - 0.1:
            torch.save(convnet.state_dict(), path)

                



if __name__ == '__main__':
    max_epochs = 100

    file_name = sys.argv[1]
    X, Y = read_data(file_name)

    labels, partition = genLabels_Partition(X, Y)
    params = {'batch_size': 4,
        'shuffle': True,
        'num_workers': 8}
    print(len(labels))
    print(len(partition['train']))

    #transform of training data
    transform = transforms.Compose([
        #transforms.Grayscale(3),
        transforms.RandomResizedCrop(size = 600, scale = (0.8, 1.0)),
        transforms.RandomRotation(20), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
        transforms.ToTensor(),
        transforms.Normalize(torch.tensor(mean), torch.tensor(std))])

    #model 
    #We use different drop out rate in different models, refer to readme.md for reference.
    linear_drop = sys.argv[3]
    dropout = sys.argv[2]
    if use_cuda:
        convnet = ConvNet2(num_classes =14, dropout = dropout, linear_drop = linear_drop).cuda()
    else:
        convnet = ConvNet2(num_classes =14, dropout = dropout, linear_drop = linear_drop)
    trainIters(
        partition = partition,
        labels = labels,
        params = params, 
        max_epochs = max_epochs,
        convnet = convnet)
