import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-7):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class CombinedLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0, weight_iou=1.0, weight_road=2.0):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.weight_iou = weight_iou
        self.weight_road = weight_road

    def forward(self, y_pred, y_true):
        # Compute BCE loss with class weights
        bce_loss = self.bce_loss(y_pred, y_true)
        
        # Compute Dice loss
        dice_loss = self.dice_coefficient_loss(y_true, y_pred)
        
        # Compute IoU loss
        iou_loss = self.iou_loss(y_true, y_pred)

        # Compute TP loss
        road_loss = self.road_loss(y_true, y_pred)
        
        # Combine losses
        loss = (self.weight_bce * bce_loss +
                self.weight_dice * dice_loss +
                self.weight_iou * iou_loss +
                self.weight_road * road_loss)/(self.weight_bce+self.weight_dice+self.weight_iou+self.weight_road)
        return loss
    
    def dice_coefficient_loss(self, y_true, y_pred, smooth=1e-6):
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        
        intersection = (y_true_flat * y_pred_flat).sum()
        dice = (2. * intersection + smooth) / (y_true_flat.sum() + y_pred_flat.sum() + smooth)
        
        return 1 - dice

    def iou_loss(self, y_true, y_pred, smooth=1e-6):
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        
        intersection = (y_true_flat * y_pred_flat).sum()
        union = y_true_flat.sum() + y_pred_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        
        return 1 - iou
    
    def road_loss(self, y_true, y_pred, smooth=1e-6):
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        
        road = (torch.sum(y_true_flat * y_pred_flat) + smooth)/(torch.sum(y_true_flat) + smooth)
        return 1 - road