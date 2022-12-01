import torch
import torch.nn as nn
import torch.nn.functional as F

class GDLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, start_with = 0):
        super(GDLoss, self).__init__()
        self.num_classes = num_classes
        self.start_with = start_with
        #Assigning more weight to left and right turns in loss calculation
        weights = [2.0,2.0,1.0] #[left, right, center]
        class_weights = torch.FloatTensor(weights)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)


    def forward(self, output, y):
        
        # print(output.shape)
        # print(y.shape)

        output = output[:, self.start_with:, :]
        y = y[:, self.start_with:]

        o1, o2, o3 = output.shape
        y1, y2 = y.shape
        
        assert(o1 == y1)
        assert(o2 == y2)
        assert(o3 == self.num_classes)
        
        op = output.reshape(o1*o2, -1)
        yp = y.reshape(o1*o2)

        loss = self.criterion(op, yp)
        return loss

        
        