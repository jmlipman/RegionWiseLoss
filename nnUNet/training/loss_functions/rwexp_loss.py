#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np
from scipy.ndimage import distance_transform_edt as dist

class BoundaryDiceLoss(nn.Module):
    def __init__(self):
        super(BoundaryDiceLoss, self).__init__()

    def forward(self, x, y_, e):
        diceloss = DiceLoss()(x, y_)
        boundaryloss = BoundaryLoss()(x, y_)
        alpha = np.max([(100 - e)/100, 0.01])

        loss = alpha*diceloss + (1-alpha)*boundaryloss

        return loss

class BoundaryLoss(nn.Module):
    # Boundary distance maps
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, x, y_):
        import nibabel as nib

        # Softmax
        x = softmax_helper(x)

        # One hot conversion
        y = torch.zeros(x.shape)
        if x.device.type == "cuda":
            y = y.cuda(x.device.index)
        y.scatter_(1, y_.long(), 1)

        y_cpu = y.detach().cpu().numpy()
        #path = "/home/miguelv/nnunet/debug/"
        #for i in range(y_cpu.shape[0]):
        #    data = np.moveaxis(y_cpu[i], 0, -1)
        #    nib.save(nib.Nifti1Image(data, np.eye(4)), path + "Y-" + str(i+1) + ".nii.gz")

        bdistmap = np.zeros_like(y_cpu)
        for b in range(bdistmap.shape[0]):
            for c in range(bdistmap.shape[1]):
                posmask = y_cpu[b, c].astype(np.bool)
                if posmask.any():
                    negmask = ~posmask
                    dist_neg = dist(negmask)
                    dist_pos = dist(posmask)

                ######
                bdistmap[b, c] = dist_neg*negmask - dist_pos*posmask

        #for i in range(bdistmap.shape[0]):
        #    data = np.moveaxis(bdistmap[i], 0, -1)
        #    nib.save(nib.Nifti1Image(data, np.eye(4)), path + "M-" + str(i+1) + ".nii.gz")

        bdistmap = torch.Tensor(bdistmap)
        if x.device.type == "cuda":
            bdistmap = bdistmap.cuda(x.device.index)

        loss = torch.mean(x * bdistmap)

        #raise Exception("llego")
        #print(loss)
        #if torch.isnan(loss):
        #    raise Exception("para")
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, x, y_):
        # Softmax
        x = softmax_helper(x)

        # One hot conversion
        y = torch.zeros(x.shape)
        if x.device.type == "cuda":
            y = y.cuda(x.device.index)
        y.scatter_(1, y_.long(), 1)

        assert (x.shape == y.shape), "x and y shapes differ: " + str(x.shape) + ", " + str(y.shape)
        axis = list([i for i in range(2, len(y.shape))]) # for 2D/3D images       
        num = 2 * torch.sum(x * y, axis=axis)                             
        denom = torch.sum(x + y, axis=axis)                               
        #print(num, denom)
        #print(num.shape, denom.shape)
        return (1 - torch.mean(num / (denom + 1e-6)))

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, x, y_):
        alpha = 1
        gamma = 2
        eps = 1e-15

        # Softmax
        x = softmax_helper(x)

        # One hot conversion
        y = torch.zeros(x.shape)
        if x.device.type == "cuda":
            y = y.cuda(x.device.index)
        y.scatter_(1, y_.long(), 1)

        ce = y * torch.log(x + eps)
        f_loss = -alpha * torch.pow(1 - x + eps, gamma) * ce
        f_loss = torch.sum(f_loss, axis=1)
        return torch.mean(f_loss)


class RWLoss(nn.Module):
    # RRW Maps
    def __init__(self):
        super(RWLoss, self).__init__()

    def forward(self, x, y_):
        import nibabel as nib

        # Softmax
        x = softmax_helper(x)

        # One hot conversion
        y = torch.zeros(x.shape)
        if x.device.type == "cuda":
            y = y.cuda(x.device.index)
        y.scatter_(1, y_.long(), 1)

        y_cpu = y.detach().cpu().numpy()
        #path = "/home/miguelv/nnunet/debug/"
        #for i in range(y_cpu.shape[0]):
        #    data = np.moveaxis(y_cpu[i], 0, -1)
        #    nib.save(nib.Nifti1Image(data, np.eye(4)), path + "Y-" + str(i+1) + ".nii.gz")

        # Probably move `y` to CPU
        rrwmap = np.zeros_like(y_cpu)
        for b in range(rrwmap.shape[0]):
            for c in range(rrwmap.shape[1]):
                rrwmap[b, c] = dist(y_cpu[b, c])
                rrwmap[b, c] = -1 * (rrwmap[b, c] / (np.max(rrwmap[b, c] + 1e-15)))
        rrwmap[rrwmap==0] = 1
        #for i in range(rrwmap.shape[0]):
        #    data = np.moveaxis(rrwmap[i], 0, -1)
        #    nib.save(nib.Nifti1Image(data, np.eye(4)), path + "M-" + str(i+1) + ".nii.gz")

        rrwmap = torch.Tensor(rrwmap)
        if x.device.type == "cuda":
            rrwmap = rrwmap.cuda(x.device.index)

        loss = torch.mean(x * rrwmap)
        #print(loss)
        #if torch.isnan(loss):
        #    raise Exception("para")
        return loss

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, x, y_):
        # Softmax
        x = softmax_helper(x)

        # One hot conversion
        y = torch.zeros(x.shape)
        if x.device.type == "cuda":
            y = y.cuda(x.device.index)
        y.scatter_(1, y_.long(), 1)

        assert (x.shape == y.shape), "x and y shapes differ: " + str(x.shape) + ", " + str(y.shape)
        ce = torch.sum(y * torch.log(x + 1e-15), axis=1)
        ce = -torch.mean(ce)
        return ce

# Losses for the last experiment
# Boundary loss
class BoundaryEqualDiceLoss(nn.Module):
    def __init__(self):
        super(BoundaryEqualDiceLoss, self).__init__()

    def forward(self, x, y_, e):
        diceloss = DiceLoss()(x, y_)
        boundaryloss = BoundaryLoss()(x, y_)

        loss = diceloss + boundaryloss

        return loss

class BoundaryEqualCELoss(nn.Module):
    def __init__(self):
        super(BoundaryEqualCELoss, self).__init__()

    def forward(self, x, y_, e):
        celoss = CELoss()(x, y_)
        boundaryloss = BoundaryLoss()(x, y_)

        loss = celoss + boundaryloss

        return loss

class BoundaryGradualCELoss(nn.Module):
    def __init__(self):
        super(BoundaryGradualCELoss, self).__init__()

    def forward(self, x, y_, e):
        celoss = CELoss()(x, y_)
        boundaryloss = BoundaryLoss()(x, y_)
        alpha = np.max([(100 - e)/100, 0.01])

        loss = alpha*celoss + (1-alpha)*boundaryloss

        return loss

# RRW
class RWEqualDiceLoss(nn.Module):
    def __init__(self):
        super(RWEqualDiceLoss, self).__init__()

    def forward(self, x, y_, e):
        diceloss = DiceLoss()(x, y_)
        rwloss = RWLoss()(x, y_)

        loss = diceloss + rwloss

        return loss

class RWEqualCELoss(nn.Module):
    def __init__(self):
        super(RWEqualCELoss, self).__init__()

    def forward(self, x, y_, e):
        celoss = CELoss()(x, y_)
        rwloss = RWLoss()(x, y_)

        loss = celoss + rwloss

        return loss

class RWGradualCELoss(nn.Module):
    def __init__(self):
        super(RWGradualCELoss, self).__init__()

    def forward(self, x, y_, e):
        celoss = CELoss()(x, y_)
        rwloss = RWLoss()(x, y_)
        alpha = np.max([(100 - e)/100, 0.01])

        loss = alpha*celoss + (1-alpha)*rwloss

        return loss

class RWGradualDiceLoss(nn.Module):
    def __init__(self):
        super(RWGradualDiceLoss, self).__init__()

    def forward(self, x, y_, e):
        diceloss = DiceLoss()(x, y_)
        rwloss = RWLoss()(x, y_)
        alpha = np.max([(100 - e)/100, 0.01])

        loss = alpha*diceloss + (1-alpha)*rwloss

        return loss
