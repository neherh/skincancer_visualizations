import torch.nn as nn

class SiameseNetwork(nn.Module):
    """Class that contains the neural network model.

    Args:
        None.

    Attributes:
        cnn1 (object): all convolutional layers
        fc1 (object):  all fully connected layers
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 3, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 3, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),

            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 1, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1*10*10, 40),
            nn.ReLU(inplace=True),

            nn.Linear(40,175), 
            nn.Sigmoid(),

            nn.Linear(175, 5))


    def forward_once(self, x):
        """Forward inference x1.

        Args:
            x (object): image tensor to be evalutated

        Returns:
            output

        """

        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output


    def forward(self, input1, input2):
        """Forward inference for two image tensors.

        Args:
            input1 (object): image tensor to be evalutated
            input2 (object): image tensor to be evalutated

        Returns:
            output1, output2

        """

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2