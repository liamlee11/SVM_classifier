# Implementation of structured SVM
import numpy as np

# A faster half-vectorized implementation.
def L_i_vectorized(x, y, W):

    ## -------------------------------------------------- ## initialization
    # Initialize values
    loss = 0.0
    dW = np.zeros(W.shape)
    num_train = x.shape[0]

    # Set hyperparameter, delta and regularization strength, lamda
    delta = 1.0
    lamda = 0.000005


    ## -------------------------------------------------- ## forward pass
    # Score function
    scores = np.dot(x, W)

    # SVM loss function and regularization
    margins = np.maximum(0, scores - scores[range(scores.shape[0]), y][..., np.newaxis] + delta)
    margins[range(scores.shape[0]), y] = 0
    loss = np.sum(margins) / num_train + lamda * np.sum(W * W)


    ## -------------------------------------------------- ## backward pass
    # Local gradient of margins
    # we go direct to dscores, so skip in here.

    # Local gradient of scores
    dscores = x

    # Calculate dW
    mask = np.zeros(margins.shape)
    mask[margins > 0] = 1
    mask[range(mask.shape[0]), y] = -1 * np.sum(margins > 0, axis = 1)
    dW = np.dot(dscores.T, mask) / num_train + 2 * lamda * W

    return loss, dW

