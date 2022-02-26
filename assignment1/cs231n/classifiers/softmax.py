from builtins import range
from click import echo_via_pager
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train=X.shape[0]
    num_classes=W.shape[1]
    for i in range(num_train):
          score=X[i].dot(W)
          
          score = np.exp(score)
          score /= np.sum(score)
          
          prob=score[y[i]]
          
          loss += np.sum(-np.log(prob))
          
          dS = score
          dS[y[i]] -= 1
          dW += np.dot(X[i].reshape(-1,1), dS.reshape(1,-1))
          
    dW /= num_train
    dW += 2*reg*W
          
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
  
def softmax_loss_vectorized(W, X, y, reg, CG=False):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    if CG : # apply computational graph
          
          X=0.01*X # preprocess to prevent overflow

          num_train=X.shape[0]
          num_classes=W.shape[1]
          score=X.dot(W)
          
          exp_correct=np.exp(score[range(num_train), y])
          exp_score_sum=np.sum(np.exp(score), axis=1)
          
          prob=exp_correct/exp_score_sum
          
          loss = np.sum(-np.log(prob))/num_train + reg * np.sum(W*W)
          
          dP=-1/(prob)
          dS = np.zeros_like(score)
          dS[:] = (((-exp_correct)/(exp_score_sum**2))*dP).reshape(-1,1)
          dS[range(num_train), y] += (1/exp_score_sum)*dP
          dW += np.dot(X.T, dS)
          dW /= num_train
          dW += 2*reg*W
    else:
          num_train=X.shape[0]

          scores=X.dot(W)
          
          scores=scores-scores.max(axis=1, keepdims=True) # to prevent overflow 
          
          scores = np.exp(scores)
          scores /= scores.sum(axis=1, keepdims=True)
          
          prob=scores[range(num_train),y]
          
          loss = np.sum(-np.log(prob))/num_train + reg * np.sum(W*W)
          
          dS = scores
          dS[range(num_train), y] -= 1
          
          dW += np.dot(X.T, dS)
          dW /= num_train
          dW += 2*reg*W
              
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
