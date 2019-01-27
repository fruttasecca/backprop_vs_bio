import torch.optim as optim
import random

"""
Backprop algorithm to be used to benchmark the training of 
neural networks, simply a wrapper for the whole process of optimizer, loss, nn model, etc.
"""


class Backprop(object):
    def __init__(self, device, model_class, model_args, lr, loss_function):
        """
        Create an instance of the algorithm, which will run the model using the SGD optimizer and
        the given lr and loss function.
        :param device: Torch device to be used.
        :param model_class:  The class of the torch network to use, so that it can be used as a constructor,
            called as model_class(**model_args).
        :param model_args: Dict containing the arguments to be used to call the constructor passed, as
            model_class(**model_args).
        :param lr: Learning rate.
        :param loss_function: Loss function with signature (X, Y) where X is the output of the network given the
        batch, Y is the predicted output to be passed to the train_step function.
        """
        self.model_args = model_args
        self.device = device
        self.model = model_class(**model_args).to(self.device)
        self.lr = lr
        self.loss_function = loss_function
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr)

    def train_step(self, batch, expected_output):
        """
        Performs a training step, given a batch of the size (batch size, data), returns the loss comparing the
        output of the model and the expected output.

        :param batch: Input batch of size (batch size, data).
        :param expected_output: Expected result of the network.
        :return: Loss computed by the loss_function passed in this instance initialization.
        Note: It returns the loss at a tuple(loss, loss, loss) just as a matter of interface
        to be easily be used in main, needs to be fixed.
        """
        # clear model gradients
        self.model.zero_grad()

        # compute results
        output = self.model.forward(batch)

        # get loss and backpropagate
        loss = self.loss_function(output, expected_output)
        loss.backward()

        # adapt weights
        self.optimizer.step()

        return loss.item(), loss.item(), loss.item()

    def to(self, device):
        """
        Sets the device of the algorithm and of the passed model.
        :param device: Pytorch like device.
        :return: Returns self
        """
        self.device = device
        self.model = self.model.to(device)
        return self

    def parameters(self):
        """
        Returns a dict mapping parameter names to values.
        :return:
        """
        res = dict()
        res["lr"] = self.lr
        res["model_parameters"] = self.model.total_parameters()
        return res

    @staticmethod
    def random_parameters():
        """
        Returns a dict of reasonable random parameters that can be used to do
        parameter search.
        :return: Dict of parameters for the algorithm.
        """
        choice = random.choice([(0.05, 0.15), (0.005, 0.015), (0.0005, 0.0015)])
        lr = random.uniform(choice[0], choice[1])
        res = dict()
        res["lr"] = lr
        return res

    @staticmethod
    def __name__():
        return "backprop"
