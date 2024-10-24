"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from a6_helper import *

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from style_transfer.py!')

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # Compute the content loss for style transfer.                       #
    ############################################################################
    # Replace "pass" statement with your code
    # pass
    # print(content_weight, content_current.shape)
    loss = content_weight * torch.sum((content_current - content_original) ** 2)
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ############################################################################
    # Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################
    # Replace "pass" statement with your code
    # pass
    features = features.reshape(features.shape[0], features.shape[1], -1) # (N, C, H*W)
    gram = torch.bmm(features, features.transpose(1,2)) # (N, C, H*W) * (N, H*W, C) -> (N, C, C)
    if normalize:
       gram /= features.shape[1] * features.shape[2]
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.   torch.Size([1, 64, 95, 127]) torch.Size([1, 128, 47, 63]) torch.Size([1, 256, 23, 31]) torch.Size([1, 256, 23, 31])
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.  [1, 4, 6, 7]
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].  torch.Size([1, 64, 64])   torch.Size([1, 128, 128])  torch.Size([1, 256, 256])  torch.Size([1, 256, 256]) 
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].  [300000, 1000, 15, 3]
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################
    # Replace "pass" statement with your code
    # pass
    target_layer = [feats[i] for i in style_layers]
    loss = 0.0
    for i in range(len(target_layer)):
       loss += style_weights[i] * torch.sum((style_targets[i] - gram_matrix(target_layer[i])) ** 2)
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    # Replace "pass" statement with your code
    # pass
    pic_1 = img[:,:,1:,:]  # H维度
    pic_2 = img[:,:,:-1,:]
    pic_3 = img[:,:,:,1:] # W维度
    pic_4 = img[:,:,:,:-1]
    loss_tv = tv_weight * (torch.sum((pic_1 - pic_2) ** 2) + torch.sum((pic_3 - pic_4) ** 2))
    return loss_tv
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def guided_gram_matrix(features, masks, normalize=True):
  """
  Inputs:
    - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
      a batch of N images.
    - masks: PyTorch Tensor of shape (N, R, H, W)
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, R, C, C) giving the
      (optionally normalized) guided Gram matrices for the N input images.
  """
  guided_gram = None
  ##############################################################################
  # Compute the guided Gram matrix from features.                        #
  # Apply the regional guidance mask to its corresponding feature and          #
  # calculate the Gram Matrix. You are allowed to use one for-loop in          #
  # this problem.                                                              #
  ##############################################################################
  # Replace "pass" statement with your code
  # pass
  masks = masks.unsqueeze(2) # (N, R, 1, H, W) 会广播
  features *= masks  
  features = features.reshape(features.shape[0], features.shape[1], features.shape[2], -1) # (N, R, C, H*W)
  guided_gram = torch.zeros((features.shape[0], features.shape[1], features.shape[2], features.shape[2]), 
                            dtype=features.dtype, device=features.device) # (N, R, C, C)
  for i in range(features.shape[1]):
    tp_guided_gram = torch.bmm(features[:,i,:,:], features[:,i,:,:].permute(0, 2, 1)) # (N, C, H*W) * (N, H*W, C) -> (N, C, C)
    guided_gram[:,i,:,:] = tp_guided_gram
  guided_gram /= features.shape[2] * features.shape[3]
  return guided_gram
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################


def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # Computes the guided style loss at a set of layers.                 #
    ############################################################################
    # Replace "pass" statement with your code
    # pass
    style_loss = 0

    for i, idx in enumerate(style_layers): # idx = 0
      style_loss += torch.sum(
         style_weights[i] * (guided_gram_matrix(feats[idx], content_masks[idx]) - style_targets[i]) ** 2
      )
    return style_loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
