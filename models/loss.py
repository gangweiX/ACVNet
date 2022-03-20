import torch.nn.functional as F
import torch


def model_loss_train_attn_only(disp_ests, disp_gt, mask):
    weights = [1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)

def model_loss_train_freeze_attn(disp_ests, disp_gt, mask):
    weights = [0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)
    
def model_loss_train(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0] 
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)
    
def model_loss_test(disp_ests, disp_gt, mask):
    weights = [1.0] 
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)
