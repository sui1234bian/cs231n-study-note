"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from abc import abstractmethod

def hello_linear_classifier():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from linear_classifier.py!')


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier(object):
  """ An abstarct class for the linear classifiers """
  # Note: We will re-use `LinearClassifier' in both SVM and Softmax
  def __init__(self):
    random.seed(0)
    torch.manual_seed(0)
    self.W = None

  def train(self, X_train, y_train, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    train_args = (self.loss, self.W, X_train, y_train, learning_rate, reg,
                  num_iters, batch_size, verbose)
    self.W, loss_history = train_linear_classifier(*train_args)
    return loss_history

  def predict(self, X):
    return predict_linear_classifier(self.W, X)

  @abstractmethod
  def loss(self, W, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative.
    Subclasses will override this.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
    - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an tensor of the same shape as W
    """
    raise NotImplementedError

  def _loss(self, X_batch, y_batch, reg):
    self.loss(self.W, X_batch, y_batch, reg)

  def save(self, path):
    torch.save({'W': self.W}, path)
    print("Saved in {}".format(path))

  def load(self, path):
    W_dict = torch.load(path, map_location='cpu')
    self.W = W_dict['W']
    print("load checkpoint file: {}".format(path))



class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """
  def loss(self, W, X_batch, y_batch, reg):
    return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """
  def loss(self, W, X_batch, y_batch, reg):
    return softmax_loss_vectorized(W, X_batch, y_batch, reg)



#**************************************************#
################## Section 1: SVM ##################
#**************************************************#

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples. When you implment the regularization over W, please DO NOT
  multiply the regularization term by 1/2 (no coefficient).

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as torch scalar
  - gradient of loss with respect to weights W; a tensor of same shape as W
  """
  dW = torch.zeros_like(W) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = W.t().mv(X[i])  # (D*C) * (D) -> (C*D) * (D) ->(C)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #######################################################################
        #                                                                #
        # Compute the gradient of the loss function and store it dW. (part 1) #
        # Rather than first computing the loss and then computing the         #
        # derivative, it is simple to compute the derivative at the same time #
        # that the loss is being computed.                                    #
        #######################################################################
        # Replace "pass" statement with your code
        # 错误的分类增加梯度 
        # 注意要除以总共的训练样本数
        dW[:, j] += X[i] / num_train  # dW:(D, C), X[i]:(D)
        # 应该正确的分类减少梯度
        dW[:, y[i]] -= X[i] / num_train
        #######################################################################
        #                       END OF YOUR CODE                              #
        #######################################################################


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * torch.sum(W * W)

  #############################################################################
  #                                                                     #
  # Compute the gradient of the loss function and store it in dW. (part 2)    #
  #############################################################################
  # Replace "pass" statement with your code
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation. When you implment
  the regularization over W, please DO NOT multiply the regularization term by
  1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as torch scalar
  - gradient of loss with respect to weights W; a tensor of same shape as W
  """
  loss = 0.0
  dW = torch.zeros_like(W) # initialize the gradient as zero (D, C)

  #############################################################################
  #                                                                      #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  # pass
  scores = torch.mm(X, W) # (N, D) * (D, C) -> (N, C)
  correct_class_score = scores[torch.arange(0, X.shape[0], 1), y].view(-1, 1) # (N,)
  margin = scores - correct_class_score + 1 # 计算loss
  margin[torch.arange(0, X.shape[0], 1), y] = 0 # 不算自己类别的损失
  margin = torch.clamp(margin, min=0) # (N, C) 将小于0的设置为0
  loss = torch.sum(margin) # loss求和 
  # loss -= torch.sum(margin[torch.arange(0, X.shape[0], 1), y]) # 减掉自己类别那部分
  loss /= X.shape[0]
  loss += reg * torch.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  #                                                                  #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # Replace "pass" statement with your code
  # pass
  # 找出大于0的部分 
  margin_mask = (margin > 0).to(X.dtype) # (N, C)
  margin_count = torch.sum(margin_mask, dim=1) # (N,) 计算dw要减几次自己 
  margin_mask[torch.arange(0, X.shape[0], 1), y] -= margin_count # 现在矩阵里面是1, 0 和-几，最多不超过-C
  dW = torch.mm(X.t(), margin_mask) # (D, N) * (N, C) -> (D, C)
  dW /= X.shape[0]
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def sample_batch(X, y, num_train, batch_size):
  """
  Sample batch_size elements from the training data and their
  corresponding labels to use in this round of gradient descent.
  """
  X_batch = None
  y_batch = None
  #########################################################################
  # Store the data in X_batch and their corresponding labels in     #
  # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
  # and y_batch should have shape (batch_size,)                           #
  #                                                                       #
  # Hint: Use torch.randint to generate indices.                          #
  #########################################################################
  # Replace "pass" statement with your code
  # pass
  idx = torch.randint(0, num_train, (batch_size,))
  X_batch = X[idx, :]
  y_batch = y[idx]
  #########################################################################
  #                       END OF YOUR CODE                                #
  #########################################################################
  return X_batch, y_batch


def train_linear_classifier(loss_func, W, X, y, learning_rate=1e-3,
                            reg=1e-5, num_iters=100, batch_size=200,
                            verbose=False):
  """
  Train this linear classifier using stochastic gradient descent.

  Inputs:
  - loss_func: loss function to use when training. It should take W, X, y
    and reg as input, and output a tuple of (loss, dW)
  - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
    classifier. If W is None then it will be initialized here.
  - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
    means that X[i] has label 0 <= c < C for C classes.
  - learning_rate: (float) learning rate for optimization.
  - reg: (float) regularization strength.
  - num_iters: (integer) number of steps to take when optimizing
  - batch_size: (integer) number of training examples to use at each step.
  - verbose: (boolean) If true, print progress during optimization.

  Returns: A tuple of:
  - W: The final value of the weight matrix and the end of optimization
  - loss_history: A list of Python scalars giving the values of the loss at each
    training iteration.
  """
  # assume y takes values 0...K-1 where K is number of classes
  num_train, dim = X.shape
  if W is None:
    # lazily initialize W
    num_classes = torch.max(y) + 1
    W = 0.000001 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)
  else:
    num_classes = W.shape[1]

  # Run stochastic gradient descent to optimize W
  loss_history = []
  for it in range(num_iters):
    # implement sample_batch function
    X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

    # evaluate loss and gradient
    loss, grad = loss_func(W, X_batch, y_batch, reg)
    loss_history.append(loss.item())

    # perform parameter update
    #########################################################################
    #                                                                  #
    # Update the weights using the gradient and the learning rate.          #
    #########################################################################
    # Replace "pass" statement with your code
    W -= learning_rate * grad
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    if verbose and it % 100 == 0:
      print('iteration %d / %d: loss %f' % (it, num_iters, loss))

  return W, loss_history


def predict_linear_classifier(W, X):
  """
  Use the trained weights of this linear classifier to predict labels for
  data points.

  Inputs:
  - W: A PyTorch tensor of shape (D, C), containing weights of a model
  - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.

  Returns:
  - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
    elemment of X. Each element of y_pred should be between 0 and C - 1.
  """
  y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
  ###########################################################################
  #                                                                   #
  # Implement this method. Store the predicted labels in y_pred.            #
  ###########################################################################
  # Replace "pass" statement with your code
  # pass
  scores = torch.mm(X, W)
  y_pred = torch.argmax(scores, dim=1)
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################
  return y_pred


def svm_get_search_params():
  """
  Return candidate hyperparameters for the SVM model. You should provide
  at least two param for each, and total grid search combinations
  should be less than 25.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  """

  learning_rates = []
  regularization_strengths = []

  ###########################################################################
  #    add your own hyper parameter lists.                             #
  ###########################################################################
  # Replace "pass" statement with your code
  # pass
  # learning_rates = [4e-2, 3e-3, 3e-2, 2e-3, 1e-3, 8e-3, 9e-4, 7e-3, 8e-4, 5e-3, 0.02, 0.01]  # 学习率候选值
  learning_rates = [4e-2, 3e-3]
  # regularization_strengths = [0.1, 1, 2.5, 2, 1.5, 0.3, 0.5, 0.7, 1.2, 0.01, 0.03, 0.05, 0.08]  # 正则化强度候选值
  regularization_strengths = [1e-1, 1e-2, 3e-2]
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################

  return learning_rates, regularization_strengths


def test_one_param_set(cls, data_dict, lr, reg, num_iters=2000):
  """
  Train a single LinearClassifier instance and return the learned instance
  with train/val accuracy.

  Inputs:
  - cls (LinearClassifier): a newly-created LinearClassifier instance.
                            Train/Validation should perform over this instance
  - data_dict (dict): a dictionary that includes
                      ['X_train', 'y_train', 'X_val', 'y_val']
                      as the keys for training a classifier
  - lr (float): learning rate parameter for training a SVM instance.
  - reg (float): a regularization weight for training a SVM instance.
  - num_iters (int, optional): a number of iterations to train

  Returns:
  - cls (LinearClassifier): a trained LinearClassifier instances with
                            (['X_train', 'y_train'], lr, reg)
                            for num_iter times.
  - train_acc (float): training accuracy of the svm_model
  - val_acc (float): validation accuracy of the svm_model
  """
  train_acc = 0.0 # The accuracy is simply the fraction of data points
  val_acc = 0.0   # that are correctly classified.
  ###########################################################################
  # TODO:                                                                   #
  # Write code that, train a linear SVM on the training set, compute its    #
  # accuracy on the training and validation sets                            #
  #                                                                         #
  # Hint: Once you are confident that your validation code works, you       #
  # should rerun the validation code with the final value for num_iters.    #
  # Before that, please test with small num_iters first                     #
  ###########################################################################
  # Feel free to uncomment this, at the very beginning,
  # and don't forget to remove this line before submitting your final version
  # num_iters = 100

  # Replace "pass" statement with your code
  # pass
  model = cls
  model.train(data_dict['X_train'], data_dict['y_train'], learning_rate=lr, reg=reg, num_iters=num_iters,
            batch_size=200, verbose=False)
  y_train_pred = model.predict(data_dict['X_train'])
  y_val_pred = model.predict(data_dict['X_val'])
  train_acc = 100.0 * (data_dict['y_train'] == y_train_pred).double().mean().item()
  val_acc = 100.0 * (data_dict['y_val'] == y_val_pred).double().mean().item()
  ############################################################################
  #                            END OF YOUR CODE                              #
  ############################################################################

  return cls, train_acc, val_acc



#**************************************************#
################ Section 2: Softmax ################
#**************************************************#

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops).  When you implment
  the regularization over W, please DO NOT multiply the regularization term by
  1/2 (no coefficient).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an tensor of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability (Check Numeric Stability #
  # in http://cs231n.github.io/linear-classify/). Plus, don't forget the      #
  # regularization!                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.  When you implment the
  regularization over W, please DO NOT multiply the regularization term by 1/2
  (no coefficient).

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability (Check Numeric Stability #
  # in http://cs231n.github.io/linear-classify/). Don't forget the            #
  # regularization!                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_get_search_params():
  """
  Return candidate hyperparameters for the Softmax model. You should provide
  at least two param for each, and total grid search combinations
  should be less than 25.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  """
  learning_rates = []
  regularization_strengths = []

  ###########################################################################
  # TODO: Add your own hyper parameter lists. This should be similar to the #
  # hyperparameters that you used for the SVM, but you may need to select   #
  # different hyperparameters to achieve good performance with the softmax  #
  # classifier.                                                             #
  ###########################################################################
  # Replace "pass" statement with your code
  pass
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################

  return learning_rates, regularization_strengths
