# ## Neural Net Definition
# We will use a standard convolutional neural network

#Imports
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 3, kernel_size=3), # 1x100x100 to 4x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3), # 4x100x100 to 8x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 4, kernel_size=3), # 8x100x100 to 8x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 3, kernel_size=3), # 8x100x100 to 4x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),

            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 1, kernel_size=3), # 4x100x100 to 1x100x100
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),
            # nn.Sigmoid(),

            # nn.MaxPool2d(4)                 # 1x100x100 to 1x25x25
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1*10*10, 40),        # 1x25x25 (625) to 500
            nn.ReLU(inplace=True),

            nn.Linear(40,175),            # 500 to 500
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),

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