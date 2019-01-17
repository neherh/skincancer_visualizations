import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """Class in which the siamese dataset is created.

    Note:
        Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Args:
        margin (int): margin or threshold to evaluate contrastive loss 

    Attributes:
        margin (int): margin or threshold to evaluate contrastive loss

    """

    def __init__(self, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """Overwritten function to eval loss.

        Args:
            output1 (object): first output of the siamese network
            output2 (object): second output of the siamese network
            label (int): label of the image

        Returns:
            lossContrastive (tensor) : returns a tensor wrapped float

        """

        euclDist = F.pairwise_distance(output1, output2)
        lossConstrastive = torch.mean((1-label) * torch.pow(euclDist, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclDist, min=0.0), 2))

        return lossConstrastive