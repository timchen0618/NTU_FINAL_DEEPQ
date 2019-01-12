import numpy as np
import pandas as pd
import torch

def read_data(file_name):
	data = pd.read_csv(file_name, encoding = "big5") 
	data = data.values
	label_only = True
	train_size = 10001
	if label_only:
		Y = data[:train_size + 1,1]
		train_X = data[:train_size + 1,0]
	else:
		Y = data[:,1]
		train_X = data[:,0]

	list_Y = []
	for y in Y:
		y = y.split(' ')
		# labels = []
		# for label in y:
		# 	labels.append(int(label))
		# 	train_Y.append(torch.tensor(labels))
		y = torch.FloatTensor(list(map(int, y)))
		y = y.unsqueeze(0)
		list_Y.append(y)


	train_Y = torch.FloatTensor((train_size, list_Y[0].size()[1]))
	torch.cat(list_Y, out = train_Y)
	train_X = np.asarray(train_X)

	return train_X, train_Y

def read_data_self_(file_name, file_name2):
	data = pd.read_csv(file_name, encoding = "big5") 
	data = data.values
	label_only = True
	train_size = 10001
	if label_only:
		Y = data[:train_size + 1,1]
		train_X = data[:train_size + 1,0]
	else:
		Y = data[:,1]
		train_X = data[:,0]

	list_Y = []
	for y in Y:
		y = y.split(' ')
		# labels = []
		# for label in y:
		# 	labels.append(int(label))
		# 	train_Y.append(torch.tensor(labels))
		y = torch.FloatTensor(list(map(int, y)))
		y = y.unsqueeze(0)
		list_Y.append(y)
        
	data2 = pd.read_csv(file_name2, encoding = "big5") 
	data2 = data2.values
    
	train_Y = torch.FloatTensor((train_size, list_Y[0].size()[1]))
	torch.cat(list_Y, out = train_Y)
	train_X = np.asarray(train_X)
    
	X2 = data2[:, 0]
	Y2 = np.asarray(data2[:, 1:], dtype = np.float32)
    
	train_X = np.concatenate((train_X, X2), axis = 0)
	print(train_X.shape)
	train_Y = torch.cat((train_Y, torch.from_numpy(Y2)), dim = 0)
	print(train_Y.size())

	return train_X, train_Y

def read_test(file_name):
	data = pd.read_csv(file_name, encoding = "big5") 
	data = data.values

	train_X = data[:,0]
	train_X = np.asarray(train_X)

	return train_X

def read_feat(file_name1, file_name2):
	data = np.load(file_name1)
	data2 = np.load(file_name2)
    
	return data, data2

def genLabels_Partition(train_X, train_Y, valid_ratio = 0.1):
	data_size = len(train_Y)
	labels = {train_X[i] : train_Y[i] for i in range(len(train_Y))}
	train_ids = [train_X[i] for i in range(int(data_size*3*valid_ratio))] + [train_X[i] for i in range(int(data_size*(4*valid_ratio)), data_size)]
	valid_ids = [train_X[i] for i in range(int(data_size*3*valid_ratio), int(data_size*4*valid_ratio))]
	partition = {'train' : train_ids, 'validation' : valid_ids}

	return labels, partition

def gen_Partition(train_X, train_Y, valid_ratio = 0.1):
	data_size = len(train_Y)
	#labels = {train_X[i] : train_Y[i] for i in range(len(train_Y))}
	train_ids = [i for i in range(int(data_size*(1-valid_ratio)), data_size)]
	valid_ids = [i for i in range(int(data_size*(1-valid_ratio)))]
	#train_ids = [i for i in range(int(data_size*(1-valid_ratio)))]
	#valid_ids = [i for i in range(int(data_size*(1-valid_ratio)), data_size)]
	partition = {'train' : train_ids, 'validation' : valid_ids}

	return partition

def genTest(train_X, valid_ratio = 0.9):
	data_size = len(train_X)
	train_ids = [train_X[i] for i in range(int(data_size))]
	valid_ids = [train_X[i] for i in range(int(data_size*(1-valid_ratio)), data_size)]
	partition = {'train' : train_ids, 'validation' : valid_ids}

	return partition

def read_data_rotate(file_name):
	data = pd.read_csv(file_name, encoding = "big5") 
	data = data.values

	Y = np.asarray(data[:,1], dtype = int)
	train_X = data[:,0]
	train_Y = torch.from_numpy(Y)
	
	return train_X, train_Y 


