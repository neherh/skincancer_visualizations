
# coding: utf-8

# # One Shot Learning with Siamese Networks
# 
# This is the jupyter notebook that accompanies



'''TO DO
    - must create new dataloader that outputs name and isn't random
    - modify test to save vector values in csv file
    - modify test to 
    - log accuracy, precision, recall(sensitivity), specificity  and whether comparison was malignant or benign and image names for test images: see https://towardsdatascience.com/pytorch-tutorial-distilled-95ce8781a89c
        - user logger and visdom
    - add visualization tools as suggested in by 'distilled' webstite and pytorch in review (pick one):
        - https://github.com/facebookresearch/visdom
        - https://github.com/lanpa/tensorboard-pytorch
    - implement command line options:
    	- test (with params like --silent and location of saving) or train
    - save all info including
        - specificity
        - sensitivity
        - accuracy
    - Image Tests to prove accuracy; 
        - method to show similarity of image (histogram of values [district or texture based ie. grouping of pixel], kertosis, std of intensities within each spatial region,  morphological analysis, connected comp analy, spatial clustering algorithm) : see https://github.com/orsinium/textdistance
        - Alex Suggests to use t-sne and some separability analysis (for the sequence)
    - Show image as a colormap for visualization purposes(Currently done matlab)
	- implement a learning rate scheduler
	- Plot ROC curve
	- save using state_dict() maybe? pytorch.org http://pytorch.org/docs/master/notes/serialization.html
'''
''' Done
    - make network input rgb images
    - Save Network weights
    - load network weights
    - pop the top of the network and input is only one image
    - show popped off layer as a grayscale sequencer
    - Save grayscale sequencer
'''

#imports
from functools import partial
import random
import sys

import click
import matplotlib		# needed for server side
matplotlib.use('Agg')	# needed for server side
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
import PIL.ImageOps
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score    
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

from siamesedataset import SiameseNetworkDataset
from siamesenetwork import SiameseNetwork
from contrastiveloss import ContrastiveLoss


# ## Helper functions
# Set of helper functions

class Config():
    training_dir =  "./skinData/train_sub_set"
    testing_dir = "./skinData/test_sub_set"
    train_batch_size =  1 #64
    train_number_epochs = 100 #d100
    saveInterval = 10

@click.command()
# @click.option('-m', '--model-pre', type=str, required=True, multiple = True)
# @click.option('-id', '--unique-id', type=str, required=True)
# @click.option('-g','--gpu', default=0)
# @click.option('-f', '--flag', type=click.Choice(['GAP','AvgRes', 'AvgSoftMax', 'SoftMax','FullFeat']), required = True)
@click.option('-c','--cuda', type = bool, default=False)
@click.option('-a', '--action', type=click.Choice(['train','test', 'testConf', 'testPR','']),
 required = True)
def main(cuda, action):

	# create dataset
	folder_dataset = dset.ImageFolder(root=Config.training_dir)
	siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
	                                        transform=transforms.Compose([transforms.Resize((10,10)),
	                                        transforms.RandomRotation((0,360)), transforms.ToTensor()]),
	                                        should_invert=False)

	# ## Training Time!
	if action == "train":

		def checkpoint(epoch):
		    model_out_path = "./modelHangar/model_epoch_{}.pth".format(epoch)
		    torch.save(net, model_out_path)
		    print("Checkpoint saved to {}".format(model_out_path))

		train_dataloader = DataLoader(siamese_dataset,
		                        shuffle=True,
		                        num_workers=8,
		                        batch_size=Config.train_batch_size)


		if cuda:
			net = SiameseNetwork().cuda()
		else:
			net = SiameseNetwork()

		criterion = ContrastiveLoss()
		optimizer = optim.Adam(net.parameters(),lr = 0.005 )
		scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 30, gamma = 0.1) 

		counter = []
		loss_history = [] 
		iteration_number= 0

		for epoch in range(0,Config.train_number_epochs):
			scheduler.step()

			for i, data in enumerate(train_dataloader,0):
				img0, img1 , label = data
				if cuda:
					img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
				else:
					img0, img1 , label = Variable(img0), Variable(img1), Variable(label)

				output1,output2 = net(img0,img1)
				optimizer.zero_grad()
				loss_contrastive = criterion(output1,output2,label)
				loss_contrastive.backward()
				optimizer.step()

				# print first loss at each epoch
				if i == 0:
					print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
					iteration_number +=10
					counter.append(iteration_number)
					loss_history.append(loss_contrastive.item())
	   
			# save model at every saveInterval
			if epoch % Config.saveInterval == 0:
				checkpoint(epoch)



	# Some simple testing
	if action == "test":
	    print('===> Loading model, popping top, and data')


	    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir, transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()]))
	    test_dataloader = DataLoader(folder_dataset_test,num_workers=6,batch_size=1,shuffle=False)

	    # if imported model that was trained using python (not python3)
	    if sys.version_info[0] < 3:
	        pretrained_model = torch.load('./trained_models/test2/model_epoch_99.pth')        # load model
	        print('in version 2 of python, encoding remains unchanged (assumes trained using python')

	    else:
	        print("in version 3 of python, changing encoding to latin (assumes trained using python")
	        pickle.load = partial(pickle.load, encoding="latin1")
	        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

	        pretrained_model = torch.load('./trained_models/test2/model_epoch_99.pth', map_location=lambda storage, loc: storage, pickle_module=pickle)        # load model
	    
		# remove fully connected layers
	    removed = list(pretrained_model.children())[:-1]
	    pretrained_model = torch.nn.Sequential(*removed)
		# print(list(pretrained_model.children()))


	    # open file
	    file = open('test.csv','ab')

	    print("===> testing the data")
	    for batch in test_dataloader:
	        x, target = Variable(batch[0]), Variable(batch[1])


	    	# generate output
	        if cuda:
	        	pretrained_model.cuda()
	        	output_final = pretrained_model.forward(x.cuda())
	        else:
	        	output_final = pretrained_model.forward_once(x)

	        npOut = output_final.cpu().data.numpy()

	        # save to csv file for visualization later
	        npOut_csv = npOut.flatten()
	        npOut_csv = np.insert(npOut_csv,0,target.data[0])
	        npOut_csv = np.reshape(npOut_csv,(1,npOut_csv.size))
	        np.savetxt(file,npOut_csv, delimiter = ",")


	# Get Confusion Matrix
	if action == "test_confusion":
	    print('===> creating datasets')

	    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir, transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()]))
	    
	    test_length = len(folder_dataset_test.imgs)

	    count_ben = 0
	    count_mal = 0
	    mal_set = set()
	    ben_set = set()

	    # get 5 unique random samples of benign and malignant
	    while True:
	        img_tuple = random.choice(folder_dataset_test.imgs)

	        if img_tuple[1] == 0:
	            if(count_ben < 4):
	                ben_set.add(img_tuple)
	                if(len(ben_set) > count_ben):
	                    count_ben += 1

	        elif img_tuple[1] == 1:
	            if(count_mal < 4):
	                mal_set.add(img_tuple)
	                if(len(mal_set) > count_mal):
	                    count_mal += 1

	        if len(ben_set) == 4 and len(mal_set) == 4:
	            break;

	    # convert to list for ease of access
	    ben_list = list(ben_set)
	    mal_list = list(mal_set)

	    # create row and col set of images
	    row_list = []
	    col_list = []
	    row_list.extend([ben_list[0][0], ben_list[1][0], mal_list[0][0], mal_list[1][0]])
	    col_list.extend([ben_list[2][0], ben_list[3][0], mal_list[2][0], mal_list[3][0]])

	    print('===> Loading model')
	    # if imported model that was trained using python (not python3)
	    if sys.version_info[0] < 3:
	        pretrained_model = torch.load('/home/vidavilane/Documents/ml_training/cancer_similarity/iteration1.pt')        # load model
	        print('in version 2 of python, encoding remains unchanged (assumes trained using python')

	    else:
	        print("in version 3 of python, changing encoding to latin (assumes trained using python")
	        pickle.load = partial(pickle.load, encoding="latin1")
	        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

	        pretrained_model = torch.load('/home/vidavilane/Documents/repos/cancer_similarity/trained_models/test0/model_epoch_0.pth', map_location=lambda storage, loc: storage, pickle_module=pickle)        # load model
	    

	    print('===> getting matrix')
	    matrix = np.zeros(shape = (4,4))
	    for i in range(0,len(row_list)):
	        for j in range(0,len(col_list)):

	            # get images
	            x0 = Image.open(row_list[i])
	            x1 = Image.open(col_list[j])

	            # transform
	            img_transform=transforms.Compose([transforms.Resize((100,100)),
	                transforms.ToTensor()
	                ])

	            # forward pass to test (put to variable, add extra dimension and convert to tensor etc)
	            if cuda:
	                output1,output2 = pretrained_model(Variable(img_transform(x0).unsqueeze(0)).cuda(),Variable(img_transform(x1).unsqueeze(0)).cuda())
	            else:
	                output1,output2 = pretrained_model(Variable(img_transform(x0).unsqueeze(0)),Variable(img_transform(x1).unsqueeze(0)))

	            # eval similarity and store
	            euclidean_distance = F.pairwise_distance(output1, output2)
	            matrix[i][j] = euclidean_distance.cpu().data.numpy()


	    print('===> save to file (csv)')

	    # save similarities in csv
	    file = open('results/confusion_mat/test_confMat.csv','wb')
	    np.savetxt(file,matrix, delimiter = ",")
	    file.close()

	    # save names in txt file
	    file = open('results/confusion_mat/test_confMat.txt','w')
	    file.write('row (top to bottom):\n')
	    file.write(row_list[0] + '\n')
	    file.write(row_list[1] + '\n')
	    file.write(row_list[2] + '\n')
	    file.write(row_list[3] + '\n\n')
	    file.write('col (left to right:\n')
	    file.write(col_list[0] + '\n')
	    file.write(col_list[1] + '\n')
	    file.write(col_list[2] + '\n')
	    file.write(col_list[3] + '\n')
	    file.close()

	# Get Precision, Recall and AP
	if action == "test_PR":

	    # varying Variables:
	    threshold = 1 # value to detemine what is same class (<=1) and what is wrong (>1)

	    print('===> Loading model')
	    # if imported model that was trained using python (not python3)
	    if sys.version_info[0] < 3:

	        #pretrained_model = torch.load('/home/vidavilane/Documents/ml_training/cancer_similarity/iteration1.pt')        # load model
		    pretrained_model = torch.load('/home/neherh/cancer_similarity/trained_models/test2/model_epoch_99.pth')        # load model
		    print('in version 2 of python, encoding remains unchanged (assumes trained using python')

	    else:
	        print("in version 3 of python, changing encoding to latin (assumes trained using python")
	        pickle.load = partial(pickle.load, encoding="latin1")
	        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
		
	    pretrained_model = torch.load('/home/neherh/cancer_similarity/trained_models/test2/model_epoch_99.pth',map_location=lambda storage, loc: storage, pickle_module=pickle)
	        # pretrained_model = torch.load('/home/vidavilane/Documents/repos/cancer_similarity/trained_models/test0/model_epoch_0.pth', map_location=lambda storage, loc: storage, pickle_module=pickle)        # load model
	    

	    print('===> Loading dataset')

	    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir, transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()]))
	    
	    test_length = len(folder_dataset_test.imgs)
	    # test_length = 3
	    print(test_length)


	    count_ben = 0
	    count_mal = 0
	    mal_set = set()
	    ben_set = set()

	    # get 5 unique random samples of benign and malignant
	    while True:
	        img_tuple = random.choice(folder_dataset_test.imgs)

	        if img_tuple[1] == 0:
	            if(count_ben < 5):
	                ben_set.add(img_tuple)
	                if(len(ben_set) > count_ben):
	                    count_ben += 1

	        elif img_tuple[1] == 1:
	            if(count_mal < 5):
	                mal_set.add(img_tuple)
	                if(len(mal_set) > count_mal):
	                    count_mal += 1

	        if len(ben_set) == 5 and len(mal_set) == 5:
	            break;

	    # convert to list for ease of access
	    ben_list = list(ben_set)
	    mal_list = list(mal_set)

	    # create row set of images
	    img_list = []
	    img_list.append(ben_list[0])
	    img_list.append(mal_list[0])
	    img_list.append(ben_list[1])
	    img_list.append(mal_list[1])
	    img_list.append(ben_list[2])
	    img_list.append(mal_list[2])
	    img_list.append(ben_list[3])
	    img_list.append(mal_list[3])
	    img_list.append(ben_list[4])
	    img_list.append(mal_list[4])

	  
	    preds  = np.zeros(shape = (test_length,len(img_list)))
	    target = np.zeros(shape = (test_length,len(img_list)))
	    # cycle through one 'class'
	    for i in range(0,len(img_list)):
	        print(i+1)
	        for j in range(0,test_length):

	            # get images
	            x0 = Image.open(img_list[i][0])###################
	            x1 = Image.open(folder_dataset_test.imgs[j][0])

	            # transform images
	            img_transform=transforms.Compose([transforms.Resize((100,100)),
	                transforms.ToTensor()
	                ])

	            #forward pass to test (put to variable, add extra dimension and convert to tensor etc)
	            if cuda:
	                pretrained_model.cuda()
	                output1,output2 = pretrained_model(Variable(img_transform(x0).unsqueeze(0)).cuda(),Variable(img_transform(x1).unsqueeze(0)).cuda())
	            else:
	                output1,output2 = pretrained_model(Variable(img_transform(x0).unsqueeze(0)),Variable(img_transform(x1).unsqueeze(0)))

	            #eval similarity and store in dissimalarity
	            euclidean_distance = F.pairwise_distance(output1, output2)
	            preds[j,i] = euclidean_distance.cpu().data.numpy() ######################
	            target[j,i] = folder_dataset_test.imgs[j][1]

	    precision = dict()
	    recall = dict()
	    average_precision = dict()
	    print('===> PR Curve and AP')
	    for i in range(len(img_list)):
	        # print(i)
	        precision[i], recall[i], _ = precision_recall_curve(target[:, i],
	                                                        preds[:, i])
	        average_precision[i] = average_precision_score(target[:, i], preds[:, i])

	    print('===> mAP')
	    # A "micro-average": quantifying score on all classes jointly
	    precision["micro"], recall["micro"], _ = precision_recall_curve(target.ravel(),
	        preds.ravel())
	    average_precision["micro"] = average_precision_score(target, preds,
	                                                        average="micro")
	    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
	        .format(average_precision["micro"]))


	    print('===> Plot')
	    plt.figure()
	    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
	             where='post')
	    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
	                     color='b')

	    plt.xlabel('Recall')
	    plt.ylabel('Precision')
	    plt.ylim([0.0, 1.05])
	    plt.xlim([0.0, 1.0])
	    plt.title(
	        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
	        .format(average_precision["micro"]))

	    plt.savefig('fig.png')


	    print('===> re-calculate mAP, started sorting')
	    val = np.array([0,1,2,3,4,5,6,7,8,9])
	    idx_preds = np.argsort(preds, axis=0)

	    new_preds = preds[idx_preds[0:10][:],val]
	    new_target = target[idx_preds[0:10][:],val]

	    print('===> re-calculate mAP from 10 rand samples, calc mAP')
	    precision = dict()
	    recall = dict()
	    average_precision = dict()
	   
	    average_precision["micro"] = average_precision_score(new_target, new_preds,
	                                                        average="micro")

	    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
	        .format(average_precision["micro"]))

if __name__ == '__main__':
	main()