"""
Implements a network visualization in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from network_visualization.py!")


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    # Hint: X.grad.data stores the gradients                                     #
    ##############################################################################
    # Replace "pass" statement with your code
    # pass
    # def forward(self, x: torch.Tensor) -> torch.Tensor: cnn的forward
        # x = self.features(x)
        # x = self.classifier(x) 
        # # self.classifier = nn.Sequential(nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        # return torch.flatten(x, 1)
    scores = model(X) # (N, Class) 1000* 
    # 不使用交叉熵的原因：计算显著性图的目的是确定输入图像的每个像素对最终分类结果的影响。因此，我们关心的是正确类别的原始分数，
    # 而不是将其转化为概率值的结果（如通过 Softmax 函数），也不需要进一步计算损失。显著性图关注的是输入特征对最终预测的影响。
    
    loss = torch.sum(scores[:, y])
    loss.backward()
    # 现在X.grad.data包含了梯度信息，形状为(N, 3, H, W)，目标形状为(N, H, W)
    X_grad_abs = torch.abs(X.grad.data) # 想要知道影响，所以计算绝对值
    # 注意要用梯度信息
    saliency = torch.max(X_grad_abs, dim=1)[0] # 计算每个channel的影响的最大值
    # (原始分数越高，loss返回的梯度绝对值就越大，反应到最终结果图上就是越红)
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return saliency


def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    """
    Generate an adversarial attack that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    - max_iter: Upper bound on number of iteration to perform
    - verbose: If True, it prints the pogress (you can use this flag for debugging)

    Returns:
    - X_adv: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our adversarial attack to the input image, and make it require
    # gradient
    X_adv = X.clone()
    X_adv = X_adv.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # Generate an adversarial attack X_adv that the model will classify    #
    # as the class target_y. You should perform gradient ascent on the score     #
    # of the target class, stopping when the model is fooled.                    #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate an adversarial     #
    # attack in fewer than 100 iterations of gradient ascent.                    #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    # Replace "pass" statement with your code
    # pass
    for i in range(max_iter):
        scores = model(X_adv) # (1, C)
        loss = torch.sum(scores[:, target_y]) ### 这一步让模型以为target_y才是真实label
        loss.backward()

        with torch.no_grad():
            ## 注意梯度是+，梯度下降是对于模型参数而言，而我们是改变图片，所以要增加梯度
            X_adv += learning_rate * (X_adv.grad / torch.norm(X_adv.grad)) ## dX = learning_rate * g / ||g||_2
            X_adv.grad.zero_()

        if verbose:
             print(f"[{i+1}] predicted class:", torch.argmax(scores, dim=1))

        if torch.argmax(scores, dim=1) == target_y:
            break
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the
    score of target_y under a pretrained model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    # Replace "pass" statement with your code
    # pass
    scores = model(img)
    loss = torch.sum(scores[:, target_y]) + l2_reg * torch.sum(img ** 2)
    loss.backward()

    with torch.no_grad():
        img += learning_rate * (img.grad / torch.norm(img.grad))
        img.grad.zero_()
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
