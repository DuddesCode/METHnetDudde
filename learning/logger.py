"""
This part of the code was originally published by mahmoodlab and adapted for this project.
https://github.com/mahmoodlab/CLAM

Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images.
Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w

"""

import numpy as np
import torch

class Accuracy_Logger(object):
    """ Class to log accuracy of network

    Attributes
    ----------
    n_classes : int
        Number of classes in classification problem
    data : list [{string : int, string : int}]

    Methods
    -------
    initialize()
        Initialize data
    log(Y_hat, Y)
        Append prediction result
    log_batch(Y_hat, Y)
        Append prediction result of batch
    get_summary(c)
        Get accuracy for class
    """
    def __init__(self, n_classes):
        """
        Parameters
        ----------
        n_classes : int
            Number of classes of classification problem to log
        """
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        """ Initialize data Attribute
        """
        self.data = [{"count": 0, "correct":0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        """ Append prediction results
        Parameters
        ----------
        Y_hat : torchTensor
            Predicted class
        Y : torchTensor
            True class
        """
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        """ Append prediction results
        Parameters
        ----------
        Y_hat : torchTensor
            Predicted classes
        Y : torchTensor
            True classes
        """
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)

        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        """ Get Accuracy and count, correct samples for class
        Parameters
        ----------
        c : int
            Class to get summary for
        Returns
        -------
        float 
            Accuracy for class
        int 
            Number of correctly predicted samples for class
        int 
            Total number of predicted samples for class
        """
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

class EarlyStopping:
    """ Class to compute wheter Early Stopping should be applied => stops if not better eresult was achieved after the current epoch

    Attributes
    ----------
    patience : int
        Number of epochs after which Early stopping is applied if not better results were achieved
    stop_epoch : int
        Minimum number of epochs to run
    verbose : bool
        Debug mode if True
    counter : int
        Current number of epochs without better result
    best_score : float
        Current best score
    early_stop : bool
        True if early stopping should trigger
    val_loss_min : float
        Best score so far

    Methods
    -------
    __call__(epoch, val_loss, model, ckpt_name='checkpoint.pt')
        Call Early stopping
    save_checkpoint(val_loss, model, ckpt_name)
        Save model
    """
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """ 
        Parameters
        ----------
        patience : int
            Number of epochs after which Early stopping is applied if not better results were achieved
        stop_epoch : int
            Minimum number of epochs to run
        verbose : bool
            True if console output is activated
        """
        self.patience = patience 
        self.stop_epoch = stop_epoch
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        """ Compute wheter Early Stopping should be applied and save model if new best score is achieved
        Parameters
        ----------
        epoch : int
            Epoch of the model training
        val_loss : float
            Current validation loss of the model
        model : AttentionMB
            Torch model trained
        ckpt_name : string
            Name of the file to save model parameters to
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        """ Save model parameters
        Parameters
        ----------
        val_loss : float
            Current validation loss of the model
        model : AttentionMB
            Torch model trained
        ckpt_name : string
            Name of the file to save model parameters to            
        """
        if self.verbose:
            print(f'Validation losss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
