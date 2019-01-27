from scipy import fftpack
import torch
import numpy as np
import random
from util import fill_weights, network_to_array


class Cosyne(object):
    def __init__(self, device, model_class, model_args, loss_function, population_size, mutation_prob, children, gamma,
                 IDCT_from=None):
        """
        Creates an instance of the Cosyne algorithm.
        :param device: Torch device to be used.
        :param model_class:  The class of the torch network to use, so that it can be used as a constructor,
            called as model_class(**model_args).
        :param model_args: Dict containing the arguments to be used to call the constructor passed, as
            model_class(**model_args).
        :param loss_function: Loss function with signature (X, Y) where X is the output of the network given the
        batch, Y is the predicted output to be passed to the train_step function.
        :param population_size: Size of the population of the Cosyne algorithm.
        :param mutation_prob: Mutation probability of each gene.
        :param children: Number of children to produce at each generation, as in the cosyne algorithm.
        :param gamma: gamma parameter of the cauchy distribution used for mutations.
        :param IDCT_from: If this value is an integer, the gene of each network will be represented by
            a vector of this length. Before using this gene to create a network an inverse discrete cosine
            transform will be applied to create the phenotype vector of the network, and from that the
            network will be "mounted"/assembled/created/ etc.
            This is a tentative to compress the network in order to reduce the search space.
            As seen in Evolving Neural Networks in Compressed Weight Space and others.
        """

        self.device = device
        # to be used to infer the number of parameters of the network (and thus the cosyne subpopulations)
        self.model = model_class(**model_args).to(device)
        self.loss_function = loss_function
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.children = children
        self.gamma = gamma
        self.IDCT_from = IDCT_from

        self.noise_generator = torch.distributions.cauchy.Cauchy(torch.tensor([0.0]).to(self.device),
                                                                 torch.tensor([self.gamma]).to(self.device))

        # length of the genes
        self.total_gene_parameters = self.model.total_parameters() if self.IDCT_from is None else self.IDCT_from

        # init population
        self.population = torch.zeros((population_size, self.total_gene_parameters), device=device,
                                      requires_grad=False)
        # init different networks form pytorch to be sure that the weight initialization is exactly
        # the same (in terms of method/algorithm)
        for i in range(population_size):
            self.model = model_class(**model_args).to(device)

            genevector = network_to_array(self.model)

            # transform using DCT if IDCT is being used
            if self.IDCT_from is not None:
                genevector = torch.tensor(
                    fftpack.dct(np.array(genevector.cpu()), n=self.total_gene_parameters, norm="ortho")).to(self.device)
            self.population[i, :] = genevector

    def train_step(self, X, Y):
        """
        Given input data and expected output of the network, perform a train step of the cosyne algorithm on the
        batch of data, essentially moving on by 1 generation.
        The loss computation is carried out using the loss_function of this object instance.
        :param X: Input data, shape of (num batches, seq len, token length)
        :param Y: Expected output data.
        :return: Best, worst and average loss obtained in this generation.
        """
        # current best result

        # evaluation step
        evaluations = self._evaluate(X, Y)
        # s = time.time()
        self._reproduce(evaluations)

        # permute each sub population
        # currently using numpy because calling torch.randperm for each dimension is way slower
        tmp = np.array(self.population.cpu())
        for i in range(self.population.size()[1]):
            tmp[:-self.children, i] = tmp[np.random.permutation(tmp.shape[0] - self.children), i]
        tmptensor = torch.from_numpy(tmp).to(self.device)
        self.population = tmptensor

        # old way of shuffling populations, slow, on cpu seems faster than gpu
        # self.population = self.population.cpu()
        # for i in range(self.population.size()[1]):
        #     self.population[:-self.children, i] = self.population[:-self.children, i][
        #         torch.randperm(self.population_size - self.children)]
        # self.population = self.population.to(self.device)

        return torch.min(evaluations).item(), torch.max(evaluations).item(), torch.mean(evaluations).item()

    def _evaluate(self, X, Y):
        """
        Evaluate the population with respect to input data X and expected output.
        :param X: Input data, shape of (num batches, seq len, token length)
        :param Y: Expected output data.
        :return: Return an array (tensor) of evaluations, 1 for each network built from the population of synapses.
        """
        # evaluate all networks
        #
        evaluations = torch.zeros(self.population_size, device=self.device)
        for i in range(self.population_size):
            # form a network by plugging the vector into the model

            genevector = self.population[i][:]

            # if IDCT is to be used first transform the vector, then use it to assemble the network
            if self.IDCT_from is not None:
                genevector = torch.tensor(
                    fftpack.idct(np.array(genevector.cpu()), n=self.model.total_parameters(), norm="ortho")).to(
                    self.device)

            fill_weights(self.model, genevector)

            # evaluate
            predicted = self.model.forward(X)
            evaluations[i] = self.loss_function(predicted, Y)
        return evaluations

    def _produce_child(self, p1, p2, i):
        """
        Given the index of parents and an index on where to place the new individual, create a new individual
        by using crossover.
        :param p1: Index of first parent.
        :param p2: Index of second.
        :param i: Index on new individual.
        """

        # randomly select genes from parents
        s1 = (torch.rand(self.total_gene_parameters, device=self.device) > 0.5).float()
        s2 = 1. - s1
        self.population[i] = s1 * self.population[p1] + s2 * self.population[p2]

    def _mutate_children(self):
        """
        Performs the mutation step on newly generated individuals.
        """
        # mutate
        # which genes are mutating
        mutation = (torch.rand((self.children, self.total_gene_parameters), device=self.device) > (
                1. - self.mutation_prob)).float()
        noise = self.noise_generator.sample([self.children, self.total_gene_parameters]).squeeze(-1)
        self.population[-self.children:, :] += noise * mutation

    def _reproduce(self, evaluations):
        """
        Given the evaluations, performs the reproduction step, creating new individuals in place of the
        worst performing ones.
        :param evaluations: Array (torch tensor) containng losses for each network.
        """
        # sort indices, get best performers (lowest error)
        sorted_evaluations = torch.argsort(evaluations)
        end_range_for_parents = max(1, int(self.population_size * 0.25))

        # create a number of children, replace worst networks
        for i in range(-(self.children - 1), 0, 2):
            # get the worst performers and replace them
            child1 = sorted_evaluations[i]
            child2 = sorted_evaluations[i - 1]

            # reproduce 2 random parents from the best 25%
            p1 = sorted_evaluations[torch.randint(0, end_range_for_parents, (1,), device=self.device)]
            p2 = sorted_evaluations[torch.randint(0, end_range_for_parents, (1,), device=self.device)]

            self._produce_child(p1, p2, child1)
            self._produce_child(p1, p2, child2)
        self._mutate_children()

    def parameters(self):
        """
        Returns a dict mapping parameter names to values.
        :return:
        """
        res = dict()
        res["population_size"] = self.population_size
        res["mutation_prob"] = self.mutation_prob
        res["children"] = self.children
        res["gamma"] = self.gamma
        res["IDCT_from"] = self.IDCT_from
        res["model_parameters"] = self.model.total_parameters()
        return res

    @staticmethod
    def random_parameters():
        """
        Returns a dict of reasonable random parameters that can be used to do
        parameter search.
        Note that no IDCT_from is specified, given that it is model (total parameters) dependant.
        :return: Dict of parameters for the algorithm.
        """
        res = dict()
        res["population_size"] = random.randrange(3, 21)
        res["mutation_prob"] = random.choice([0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])

        children = random.choice(range(2, res["population_size"], 2))
        res["children"] = children
        res["gamma"] = random.uniform(0.05, 0.4)
        return res

    @staticmethod
    def __name__():
        return "cosyne"
