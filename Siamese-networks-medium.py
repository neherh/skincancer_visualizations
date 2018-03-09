
# coding: utf-8

# # One Shot Learning with Siamese Networks
# 
# This is the jupyter notebook that accompanies



'''TO DO
    - Plot ROC curve
    - save all info including
        - specificity
        - sensitivity
        - accuracy
    - command line options
    - Image Tests to prove accuracy
    - Show image as a colormap for visualization purposes(Currently done matlab)
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
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


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
    training_dir = "/home/jzelek/Documents/datasets/SkinData/train_sub_set/"
    testing_dir = "/home/jzelek/Documents/datasets/SkinData/test_sub_set/"
    train_batch_size =  20 #64
    train_number_epochs = 5 #d100


# ## Custom Dataset Class
# This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair
class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
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


# ## Using Image Folder Dataset

folder_dataset = dset.ImageFolder(root=Config.training_dir)



siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Scale((100,100)),
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
            nn.Conv2d(3, 4, kernel_size=3), # 1x100x100 to 4x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3), # 4x100x100 to 8x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3), # 8x100x100 to 8x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 4, kernel_size=3), # 8x100x100 to 4x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 1, kernel_size=3), # 4x100x100 to 1x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),

            nn.MaxPool2d(4)                 # 1x100x100 to 1x25x25
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1*25*25, 500),        # 1x25x25 (625) to 500
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),            # 500 to 500
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))              # 500 to 5

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


# ## Training Time!

train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.train_batch_size)


net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

counter = []
loss_history = [] 
iteration_number= 0


for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
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
show_plot(counter,loss_history)

print('save model')
torch.save(net,'final_training.pt')


# ## Some simple testing
# The last 3 subjects were held out from the training, and will be used to test. The Distance between each image pair denotes the degree of similarity the model found between the two images. Less means it found more similar, while higher values indicate it found them to be dissimilar.

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Scale((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)

for i in range(1):
    _,x1,label2 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    
    output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)

    if label2[0][0] == 0:
        val = 'dissimilar'
    else:
        val = 'similar'
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]), 'moles are ' + val)



# load new model, pop top, run through image, show image
print('loading model, popping top, running through img, showing image')
pretrained_model = torch.load('final_training.pt')        # load model

# remove fully connected layers
removed = list(pretrained_model.children())[:-1]
pretrained_model = torch.nn.Sequential(*removed)
# print(list(pretrained_model.children()))

# generate output
output_final = pretrained_model.forward(Variable(x1).cuda())
# output_final.norm()
npOut = output_final.cpu().data.numpy()
# npOut.reshape((1,100,100,1))
# npOut = npOut[0]
# npOut = npOut[0][0]
print(type(npOut))
print(npOut.size)
print(npOut[0][0].ndim)
npOut *= 255.0/npOut.max()
imgOut = Image.fromarray(npOut[0][0],'L')
imgOut.save('my.png')
imgOut.show()
# normOut = normalize(npOut[:,np.newaxis], axis = 0).ravel()
# print(normout)
# print(len(output_final))
# print(type(output_final))
# print(output_final)