import numpy as np
import pandas as pd
import sys

def read_data(file_name, out_name):
	data = pd.read_csv(file_name, encoding = "big5")
	train_size = 10001 
	data = data[:train_size + 1]
	data = data.sample(frac=1).reset_index(drop=True)
	data.to_csv(out_name, index = False)

	# data = data.values
	# label_only = True
	
	# if label_only:
	# 	Y = data[:train_size + 1,1]
	# 	train_X = data[:train_size + 1,0]
	# else:
	# 	Y = data[:,1]
	# 	train_X = data[:,0]

	# list_Y = []
	# for y in Y:
	# 	y = y.split(' ')
	# 	# labels = []
	# 	# for label in y:
	# 	# 	labels.append(int(label))
	# 	# 	train_Y.append(torch.tensor(labels))
	# 	y = torch.FloatTensor(list(map(int, y)))
	# 	y = y.unsqueeze(0)
	# 	list_Y.append(y)


	# train_Y = torch.FloatTensor((train_size, list_Y[0].size()[1]))
	# torch.cat(list_Y, out = train_Y)
	# train_X = np.asarray(train_X)

	# return train_X, train_Y

file_name = sys.argv[1]
out_name = "label_only.csv"
read_data(file_name, out_name)