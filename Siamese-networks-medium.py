
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

#Imports
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
from sklearn.preprocessing import normalize
import random
from PIL import Image
import torch
print(torch.__version__)
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from functools import partial
import pickle

import sys

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


import argparse

# command line to get required actions
parser = argparse.ArgumentParser()
parser.add_argument("action")

args = parser.parse_args()

print(args.action)

#############################################
# global variables
isCuda = False
#############################################

# ## Helper functions
# Set of helper functions

def imshow(img,text=None, diagnose = None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    if diagnose:
        plt.text(75, 0, diagnose, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


def save_checkpoint(state, is_final = False, filename = 'checkpoint.pth.tar'):
    if is_final:
        torch.save(state,'final_weights.pth.tar')
    else:
        torch.save(state,filename)


# ## Configuration Class
# A simple class to manage configuration

class Config():
    ################ -torchz- #######################
    # training_dir =  "/home/neherh/train_set_cropped"
    # testing_dir = "/home/neherh/test_set_cropped" # "/home/jzelek/Documents/datasets/SkinData/train_sub_set/"
    
    ################ -vidavilane- #######################
    training_dir =  "/home/vidavilane/Documents/repos/cancer_similarity/SkinData/train_sub_set"
    testing_dir =   "/home/vidavilane/Documents/repos/cancer_similarity/SkinData/test_sub_set" # "/home/jzelek/Documents/datasets/SkinData/test_sub_set/"
    train_batch_size =  12 #64
    train_number_epochs = 100 #d100


# ## Custom Dataset Class
# This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair
class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True, randomize = True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        self.randomize = randomize
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)



folder_dataset = dset.ImageFolder(root=Config.training_dir)



siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                        							  transforms.RandomRotation((0,360)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)


# ## Visualising some of the data
# The top row and the bottom row of any column is one pair. The 0s and 1s correspond to the column of the image.
# 0 indiciates dissimilar, and 1 indicates similar.

# vis_dataloader = DataLoader(siamese_dataset,
#                         shuffle=True,
#                         num_workers=8,
#                         batch_size=8)
# dataiter = iter(vis_dataloader)


# example_batch = next(dataiter)
# concatenated = torch.cat((example_batch[0],example_batch[1]),0)
# imshow(torchvision.utils.make_grid(concatenated))
# print(example_batch[2].numpy())


# ## Neural Net Definition
# We will use a standard convolutional neural network

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, kernel_size=3), # 1x100x100 to 4x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, kernel_size=3), # 4x100x100 to 8x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3), # 8x100x100 to 8x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size=3), # 8x100x100 to 4x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 1, kernel_size=3), # 4x100x100 to 1x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),

            nn.MaxPool2d(4)                 # 1x100x100 to 1x25x25
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1*25*25, 400),        # 1x25x25 (625) to 500
            nn.ReLU(inplace=True),

            nn.Linear(400,175),            # 500 to 500
            nn.ReLU(inplace=True),

            nn.Linear(175, 5))              # 500 to 5

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# ## Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

# ## Training Time!
if args.action == "train":

	train_dataloader = DataLoader(siamese_dataset,
	                        shuffle=True,
	                        num_workers=8,
	                        batch_size=Config.train_batch_size)


	if isCuda:
		net = SiameseNetwork().cuda()
	else:
		net = SiameseNetwork()

	criterion = ContrastiveLoss()
	optimizer = optim.Adam(net.parameters(),lr = 0.005 )
	scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 30, gamma = 0.1) 

	counter = []
	loss_history = [] 
	iteration_number= 0


	cnt = 0
	for epoch in range(0,Config.train_number_epochs):
		scheduler.step()

		for i, data in enumerate(train_dataloader,0):
			img0, img1 , label = data
			if isCuda:
				img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
			else:
				img0, img1 , label = Variable(img0), Variable(img1), Variable(label)

			output1,output2 = net(img0,img1)
			optimizer.zero_grad()
			loss_contrastive = criterion(output1,output2,label)
			loss_contrastive.backward()
			optimizer.step()
			if i %10 == 0 :
				print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
				iteration_number +=10
				counter.append(iteration_number)
				loss_history.append(loss_contrastive.data[0])
   
                # save model every 5 epochs
		cnt+=1

		if cnt ==1 or epoch == Config.train_batch_size:
			checkpoint(epoch)
			cnt = 0


	# #show_plot(counter,loss_history)

	# print('save model')
	# torch.save(net,'final_training.pt')


# ## Some simple testing
if args.action == "test":
    print('===> Loading model, popping top, and data')


    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir, transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()]))
    test_dataloader = DataLoader(folder_dataset_test,num_workers=6,batch_size=1,shuffle=False)



	# load new model, pop top, run through image, show image
	# print('loading model, popping top, running through img, showing image')


    # if imported model that was trained using python (not python3)
    if sys.version_info[0] < 3:
        pretrained_model = torch.load('/home/vidavilane/Documents/ml_training/cancer_similarity/iteration1.pt')        # load model
        print('in version 2 of python, encoding remains unchanged (assumes trained using python')

    else:
        print("in version 3 of python, changing encoding to latin (assumes trained using python")
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

        pretrained_model = torch.load('/home/vidavilane/Documents/ml_training/cancer_similarity/iteration1.pt', map_location=lambda storage, loc: storage, pickle_module=pickle)        # load model
    


	# remove fully connected layers
    # removed = list(pretrained_model.children())[:-1]
    # pretrained_model = torch.nn.Sequential(*removed)
	# print(list(pretrained_model.children()))


    # open file
    file = open('test.csv','ab')

    print("===> testing the data")
    for batch in test_dataloader:
        x, target = Variable(batch[0]), Variable(batch[1])


    	# generate output
        if isCuda:
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
if args.action == "test_confusion":
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

    # create row set of images
    row_list = []
    row_list.append(ben_list[0][0])
    row_list.append(ben_list[1][0])
    row_list.append(mal_list[0][0])
    row_list.append(mal_list[1][0])

    # create column set of images
    col_list = []
    col_list.append(ben_list[2][0])
    col_list.append(ben_list[3][0])
    col_list.append(mal_list[2][0])
    col_list.append(mal_list[3][0])


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
            if isCuda:
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
if args.action == "test_PR":

    # varying Variables:
    threshold = 1 # value to detemine what is same class (<=1) and what is wrong (>1)

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
    

    print('===> Loading dataset')

    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir, transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()]))
    
    test_length = len(folder_dataset_test.imgs)
    # test_length = 3
    print(test_length)
  
    preds  = np.zeros(shape = (test_length,test_length))
    target = np.zeros(shape = (test_length,test_length))
    # cycle through one 'class'
    for i in range(0,test_length):
        for j in range(0,test_length):


            # get images
            x0 = Image.open(folder_dataset_test.imgs[i][0])###################
            x1 = Image.open(folder_dataset_test.imgs[j][0])

            # transform images
            img_transform=transforms.Compose([transforms.Resize((100,100)),
                transforms.ToTensor()
                ])

            #forward pass to test (put to variable, add extra dimension and convert to tensor etc)
            if isCuda:
                output1,output2 = pretrained_model(Variable(img_transform(x0).unsqueeze(0)).cuda(),Variable(img_transform(x1).unsqueeze(0)).cuda())
            else:
                output1,output2 = pretrained_model(Variable(img_transform(x0).unsqueeze(0)),Variable(img_transform(x1).unsqueeze(0)))

            #eval similarity and store in dissimalarity
            euclidean_distance = F.pairwise_distance(output1, output2)
            preds[j,i] = euclidean_distance.cpu().data.numpy() ######################
            target[j,i] = folder_dataset_test.imgs[j][1]
        # print(folder_dataset_test.imgs[j][1])
        # store target value 
    # print(target.size)

    # for i in range(0,3):

    # print(target[:,0])

    # For each class

    # print(target)
    # print(preds)

    # normalize preds in current setup
    # preds = preds/preds.max(axis = 0)

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(test_length):
        print(i)
        precision[i], recall[i], _ = precision_recall_curve(target[:, i],
                                                        preds[:, i])


    # print(precision)
    # print(recall)
        average_precision[i] = average_precision_score(target[:, i], preds[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(target.ravel(),
        preds.ravel())
    average_precision["micro"] = average_precision_score(target, preds,
                                                        average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
        .format(average_precision["micro"]))



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

    plt.savefig('fig.jpg')
    # plt.show()