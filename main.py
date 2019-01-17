# -*- coding: utf-8 -*-
"""main.py

This module trains or tests a neural network model based on the parameters set.
Some testing includes, saving into csv, computer confusion matrix, and 
computing precision and recall.

Examples:
    To train::

        $ python example_google.py

    To test::
       $ python main.py --test getCSV -m ./modelHangar/model_epoch_1.pth

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * include t-sne evaluation

"""

from functools import partial
import random
import sys

import click
import matplotlib     # needed for server side
matplotlib.use('Agg')   # needed for server side
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

from contrastiveloss import ContrastiveLoss
from siamesedataset import SiameseNetworkDataset
from siamesenetwork import SiameseNetwork


class Config():
    """Configuration which holds important variables.

    Args:
        None.

    Attributes:
        trainingDir (str): training directory
        testingDir (str): testing directory
        trainBatchSize (int): batch size for training
        trainNumberEpochs (int): num of epochs for training
        saveInterval (int): interval for saving model
    """
    trainingDir =  "./skinData/train_sub_set"
    testingDir = "./skinData/test_sub_set"
    trainBatchSize =  1
    trainNumberEpochs = 100
    saveInterval = 10
    imageSize = 10 # hxw = 10x10, pixels


class NotRequiredIf(click.Option):
    """Class to remove args when unneeded.

    Note:
       Inspired from https://stackoverflow.com/questions/
       44247099/click-command-line-interfaces-make-options-
       required-if-other-optional-option-is

    Args:
        click.option (object): containing arg options

    """

    def __init__(self, *args, **kwargs):
        self.not_required_if = kwargs.pop('not_required_if')
        assert self.not_required_if, "'not_required_if' parameter required"
        kwargs['help'] = (kwargs.get('help', '') +
            ' NOTE: This argument is mutually exclusive with %s' %
            self.not_required_if
        ).strip()
        super(NotRequiredIf, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        """overridden function, which handles and parses result.

        Args:
            ctx (object): 
            opts (list): option list
            args (list): args list

        Attr:
            weArePresent (str)
            otherPresent (str)

        Returns:
            output

        """
    
        weArePresent = self.name in opts
        otherPresent = self.not_required_if in opts

        '''
        Determines if some options are not needed in the 
        arg parsing. If desired to be ignored, it will return
        a none value.
        '''
        if otherPresent:
            if weArePresent:
                raise click.UsageError(
                    "Illegal usage: `%s` is mutually exclusive with `%s`" % (
                        self.name, self.not_required_if))
            else:
                self.prompt = None

        return super(NotRequiredIf, self).handle_parse_result(
            ctx, opts, args)


# refer to NotRequiredIf Class for details
@click.command()
@click.option('-c','--cuda', type = bool, default=False)
@click.option('--train', prompt = True, is_flag=True, cls=NotRequiredIf, not_required_if = 'test')
@click.option('--test', prompt = True, type=click.Choice(['getCSV', 'getConf', 'getPR','']),
    cls=NotRequiredIf, not_required_if = 'train')
@click.option('-m', '--model', prompt = True, type=str, cls=NotRequiredIf, not_required_if = 'train')

def main(cuda, train, test, model):
    """main function to train and test the neural network.

    Args:
        cuda (bool): is cuda needed? 
        train (bool): flag is by default true 
        test (object): click choice option  
        model (str): model file path for loading

    Attr:
        cuda (bool): is cuda needed? 
        train (bool): flag is by default true 
        test (object): click choice option  
        model (str): model file path for loading

    Returns:
        None.

    """

    if train:

        def train_network():
            """Trains the network.

            Attr:
                counter:
                criterion:
                epoch:
                folderDataset:
                iterationNumber:
                lossContrastive:
                lossHistory:
                modelOutpath:
                net:
                optimizer:
                scheduler:
                siameseDataset:
                trainDataLoader:

            Returns:
                None.

            """
       
            def checkpoint(epoch):
                """checkpoint to save model

                Args:
                    epoch (int): number of epoch

                Attr:
                    modelOutPath: file path of outputted model

                Returns:
                    None.

                """

                modelOutPath = "./modelHangar/model_epoch_{}.pth".format(epoch)
                torch.save(net, modelOutPath)
                print("Checkpoint saved to {}".format(modelOutPath))

            # create dataset and dataloader
            folderDataset = dset.ImageFolder(root=Config.trainingDir)
            siameseDataset = SiameseNetworkDataset(imageFolderDataset=folderDataset,
                transform=transforms.Compose([transforms.Resize((10, 10)), transforms.RandomRotation((0, 360)),
                transforms.ToTensor()]), shouldInvert=False)
            trainDataloader = DataLoader(siameseDataset, shuffle=True, num_workers=8,
                batch_size=Config.trainBatchSize)

            if cuda: # init network, dependent on gpu
                net = SiameseNetwork().cuda()
            else:
                net = SiameseNetwork()

            # init training params
            criterion = ContrastiveLoss()
            optimizer = optim.Adam(net.parameters(), lr = 0.005 )
            scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 30, gamma = 0.1) 

            counter = []
            lossHistory = [] 
            iterationNumber = 0

            # train model
            for epoch in range(0, Config.trainNumberEpochs):
                scheduler.step()

                for i, data in enumerate(trainDataloader, 0):
                    img0, img1 , label = data
                    if cuda:
                        img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
                    else:
                        img0, img1 , label = Variable(img0), Variable(img1), Variable(label)

                    output1,output2 = net(img0,img1)
                    optimizer.zero_grad()
                    lossContrastive = criterion(output1, output2, label)
                    lossContrastive.backward()
                    optimizer.step()

                    if i == 0: # print first loss at each epoch
                        print("Epoch number {}\n Current loss {}\n".format(epoch,lossContrastive.item()))
                        iterationNumber += 10
                        counter.append(iterationNumber)
                        lossHistory.append(lossContrastive.item())
          
                # save model at every saveInterval
                if epoch % Config.saveInterval == 0:
                    checkpoint(epoch)

        train_network()

    if test is not None: # test to get csv, confusion matrix or precision/recall

        pretrainedModel = None
        folderDatasetTest = dset.ImageFolder(root=Config.testingDir, 
           transform = transforms.Compose([transforms.Resize((Config.imageSize, Config.imageSize)), transforms.ToTensor()]))

        if sys.version_info[0] < 3: # imported model trained using python2.7 than 3.6 changes encoding
            pretrainedModel = torch.load(model)
            print('in version 2 of python, encoding remains unchanged (assumes trained using python)')

        else:
            print("in version 3 of python, changing encoding to latin (assumes trained using python)")
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            pretrainedModel = torch.load(model, map_location=lambda storage, loc: storage, pickle_module=pickle) # load model
        
        if test == 'getCSV':

            def get_csv(pretrainedModel):
                """Trains the network.

                Args:
                    pretrainedModel (object):

                Attr:
                    file:
                    npOut:
                    npOutCSV
                    outputFinal:
                    pretrainedModel (object):
                    removed (object): tensor of removed fc layers
                    testDataLoader:

                Returns:
                    None.

                """

                # load data and pop top of network
                testDataloader = DataLoader(folderDatasetTest, num_workers=6, batch_size=1, shuffle=False)    
                removed = list(pretrainedModel.children())[:-1]
                pretrainedModel = torch.nn.Sequential(*removed)


                # open file, then save in csv file
                print("===> Testing data, then saving")
                file = open('./results/test.csv','ab')
                for batch in testDataloader:
                    x, target = Variable(batch[0]), Variable(batch[1])
                    if cuda:
                        pretrainedModel.cuda()
                        outputFinal = pretrainedModel.forward(x.cuda())
                    else:
                        outputFinal = pretrainedModel.forward(x)

                    npOut = outputFinal.cpu().data.numpy()
                    npOutCSV = npOut.flatten()
                    npOutCSV = np.insert(npOutCSV,0,target.data[0])
                    npOutCSV = np.reshape(npOutCSV,(1,npOutCSV.size))
                    np.savetxt(file,npOutCSV, delimiter = ",")

                print('Saved results in ./results/test.csv')

            get_csv(pretrainedModel)

        # Get Confusion Matrix
        elif test == "getConf":

            def get_confusion_matrix():
                """Trains the network.

                Attr:
                    countBen (int):
                    countMal (int):
                    benSet (set):
                    malSet (set):
                    imgTuple (tuple):
                    benList (list):
                    malList (list):
                    rowList (list):
                    colList (list):
                    matrix (object): confusion matrix
                    euclDist (float): euclidean distance from images 
                    file (object): csv file of confusion matrix 

                Returns:
                    None.

                """
                countBen = 0
                countMal = 0
                malSet = set()
                benSet = set()

                # get 5 unique random samples of benign and malignant
                while True:
                    imgTuple = random.choice(folderDatasetTest.imgs)

                    if imgTuple[1] == 0: # benign
                        if(countBen < 4):
                            benSet.add(imgTuple)
                            if(len(benSet) > countBen):
                                countBen += 1

                    elif imgTuple[1] == 1: # malignant
                        if(countMal < 4):
                            malSet.add(imgTuple)
                            if(len(malSet) > countMal):
                                countMal += 1

                    if len(benSet) == 4 and len(malSet) == 4:
                        break;

                benList = list(benSet) # convert to list for ease of access
                malList = list(malSet)

                # create row and col set of images
                rowList = []
                colList = []
                rowList.extend([benList[0][0], benList[1][0], malList[0][0], malList[1][0]])
                colList.extend([benList[2][0], benList[3][0], malList[2][0], malList[3][0]])

                # get calculate confusion matrix
                matrix = np.zeros(shape = (4,4))
                for i in range(0,len(rowList)):
                    for j in range(0,len(colList)):
                        # get images
                        x0 = Image.open(rowList[i])
                        x1 = Image.open(colList[j])

                        # transform
                        imgTransform=transforms.Compose([transforms.Resize((Config.imageSize, Config.imageSize)),
                            transforms.ToTensor()
                            ])

                        # forward pass to test (put to variable, add extra dimension and convert to tensor etc)
                        if cuda:
                            output1,output2 = pretrainedModel(Variable(imgTransform(x0).unsqueeze(0)).cuda(),
                                Variable(imgTransform(x1).unsqueeze(0)).cuda())
                        else:
                            output1,output2 = pretrainedModel(Variable(imgTransform(x0).unsqueeze(0)),
                                Variable(imgTransform(x1).unsqueeze(0)))

                        # eval similarity and store
                        euclDist = F.pairwise_distance(output1, output2)
                        matrix[i][j] = euclDist.cpu().data.numpy()

                # save similarities in csv
                file = open('./results/test_confMat.csv', 'wb')
                np.savetxt(file, matrix, delimiter = ",")
                file.close()

                # save names in txt file
                file = open('./results/test_confMat.txt', 'w')
                file.write('row (top to bottom):\n')
                file.write(rowList[0] + '\n')
                file.write(rowList[1] + '\n')
                file.write(rowList[2] + '\n')
                file.write(rowList[3] + '\n\n')
                file.write('col (left to right:\n')
                file.write(colList[0] + '\n')
                file.write(colList[1] + '\n')
                file.write(colList[2] + '\n')
                file.write(colList[3] + '\n')
                file.close()

                print('===> saved to file to ./results/test_confMat.csv')

            get_confusion_matrix()

        elif test == "getPR": # Get Precision, Recall and AP

            def get_precision_recall():
                """Trains the network.

                Attr:
                    testLength:
                    countBen (int):
                    countMal (int):
                    benSet (set):
                    malSet (set):
                    imgTuple (tuple):
                    benList (list):
                    malList (list):
                    imgList (list):
                    preds(object): np array of predictions
                    target(object): np arracy of target (true values)
                    euclDist (float): euclidean distance from images 
                    precision (dict):
                    recall (dict):
                    avgPrecision (dict):
                    file (object): csv file of confusion matrix 

                Returns:
                    None.

                """

                # varying Variables:
                testLength = len(folderDatasetTest.imgs)
                countBen = 0
                countMal = 0
                malSet = set()
                benSet = set()

                # get 5 unique random samples of benign and malignant
                while True:
                    imgTuple = random.choice(folderDatasetTest.imgs)
                    if imgTuple[1] == 0:
                        if(countBen < 5):
                            benSet.add(imgTuple)
                            if(len(benSet) > countBen):
                                countBen += 1

                    elif imgTuple[1] == 1:
                        if(countMal < 5):
                            malSet.add(imgTuple)
                            if(len(malSet) > countMal):
                                countMal += 1

                    if len(benSet) == 5 and len(malSet) == 5:
                        break;

                # convert to list for ease of access
                benList = list(benSet)
                malList = list(malSet)

                # create row set of images
                imgList = []
                imgList.extend([benList[0], malList[0], benList[1], malList[1], benList[2], malList[2],
                    benList[3], malList[3], benList[4], malList[4]])
                        
                preds  = np.zeros(shape = (testLength, len(imgList)))
                target = np.zeros(shape = (testLength, len(imgList)))

                # cycle through one 'class'
                for i in range(0,len(imgList)):
                    print(i+1)
                    for j in range(0,testLength):

                        # get images
                        x0 = Image.open(imgList[i][0])
                        x1 = Image.open(folderDatasetTest.imgs[j][0])

                        # transform images
                        imgTransform=transforms.Compose([transforms.Resize((Config.imageSize, Config.imageSize)),
                            transforms.ToTensor()
                            ])

                        # forward pass to test (put to variable, add extra dimension and convert to tensor etc)
                        if cuda:
                            pretrainedModel.cuda()
                            output1,output2 = pretrainedModel(Variable(imgTransform(x0).unsqueeze(0)).cuda(),
                                Variable(imgTransform(x1).unsqueeze(0)).cuda())
                        else:
                            output1,output2 = pretrainedModel(Variable(imgTransform(x0).unsqueeze(0)),
                                Variable(imgTransform(x1).unsqueeze(0)))

                        # eval similarity and store in dissimilarity
                        euclDist = F.pairwise_distance(output1, output2)
                        preds[j,i] = euclDist.cpu().data.numpy()
                        target[j,i] = folderDatasetTest.imgs[j][1]

                precision = dict()
                recall = dict()
                avgPrecision = dict()

                print('===> PR Curve and AP')
                for i in range(len(imgList)):
                    precision[i], recall[i], _ = precision_recall_curve(target[:, i], preds[:, i])
                    avgPrecision[i] = avgPrecision_score(target[:, i], preds[:, i])

                print('===> mAP')
                # A "micro-average": quantifying score on all classes jointly
                precision["micro"], recall["micro"], _ = precision_recall_curve(target.ravel(),
                    preds.ravel())
                avgPrecision["micro"] = average_precision_score(target, preds, average="micro")
                print('Average precision score, micro-averaged over all classes: {0:0.2f}'
                    .format(avgPrecision["micro"]))

                print('===> Plot')
                plt.figure()
                plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
                plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title(
                    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                    .format(avgPrecision["micro"]))

                plt.savefig('fig.png')

                print('===> re-calculate mAP, started sorting')
                val = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                idxPreds = np.argsort(preds, axis=0)

                newPreds = preds[idxPreds[0:10][:], val]
                newTarget = target[idxPreds[0:10][:], val]

                print('===> re-calculate mAP from 10 rand samples, calc mAP')
                avgPrecision = dict()
                avgPrecision["micro"] = average_precision_score(newTarget, newPreds,
                    average="micro")

                print('Average precision score, micro-averaged over all classes: {0:0.2f}'
                    .format(avgPrecision["micro"]))

            get_precision_recall()

if __name__ == '__main__':
    main()