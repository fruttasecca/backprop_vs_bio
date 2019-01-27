#!/usr/bin/python3

import torch
import random
from util import fill_weights, network_to_array
from scipy import fftpack
import numpy as np


class EA(object):

    def __init__(self, device, model_class, model_args, loss_function, population_size, mutation_prob,
                 crossover, selection, sigma, crossover_method, selection_method, best_rate,
                 n_parents, elitism, IDCT_from=None):
        """
        Creates an instance of Evolutionary algorithm.

        :param device: Torch device to be used.
        :param model_class:  The class of the torch network to use, so that it can be used as a constructor,
            called as model_class(**model_args).
        :param model_args: Dict containing the arguments to be used to call the constructor passed, as
            model_class(**model_args).
        :param loss_function: fitness function
        :param population_size: size of the generated population
        :param mutation_prob: probability of each child of being mutated
        :param crossover: apply or not crossover, boolean
        :param selection: apply or not selection, boolean
        :param sigma: mutation parameter sigma
        :param crossover_method: crossover method to be applied,
            possible methods are: "single_swap", "uniform_swap", "arithmetic"
        :param selection_method: selection method, possible methods are: "truncated", "fitness_based", "rank_based"
        :param best_rate: rate of best population for the selection method
        :param n_parents: number of parents for crossover operator
        :param IDCT_from: If this value is an integer, the gene of each network will be represented by
            a vector of this length. Before using this gene to create a network an inverse discrete cosine
            transform will be applied to create the phenotype vector of the network, and from that the
            network will be "mounted"/assembled/created/ etc.
            This is a tentative to compress the network in order to reduce the search space.
            As seen in Evolving Neural Networks in Compressed Weight Space and others.
        """
        self.population_size = population_size
        self.device = device
        self.loss_function = loss_function
        self.mutation_prob = mutation_prob
        self.crossover = crossover
        self.selection = selection
        self.sigma = sigma
        self.crossover_method = crossover_method
        self.selection_method = selection_method
        self.best_rate = best_rate
        self.n_parents = n_parents
        self.IDCT_from = IDCT_from
        self.elitism = elitism

        # elitism , 1 best individual kept
        self.best_eval = None
        self.best_individual = None

        # population initialization : starting from pytorch initializat ion
        model = model_class(**model_args).to(device)

        # length of the genes
        self.total_gene_parameters = model.total_parameters() if self.IDCT_from is None else self.IDCT_from

        self.genlength = len(network_to_array(model))
        if self.IDCT_from is not None:
            self.genlength = self.IDCT_from
        self.population = torch.zeros((population_size, self.genlength), device=self.device)
        for i in range(population_size):
            model = model_class(**model_args).to(device)
            # transform using DCT if IDCT is being used
            genevector = network_to_array(model).cpu()
            if self.IDCT_from is not None:
                genevector = torch.tensor(
                    fftpack.dct(np.array(genevector), n=self.total_gene_parameters, norm="ortho"))
            self.population[i] = genevector.to(device)

        # self.population =self.population.cpu()
        self.model = model

    def train_step(self, X, Y):
        # evaluate all the individuals
        evaluations = self._evaluate(X, Y)
        # noise generator for mutation
        noise_generator = torch.distributions.normal.Normal(torch.tensor([0.0], device=self.device),
                                                            torch.tensor([1.0], device=self.device))
        best = self._selection(evaluations, selection=self.selection,
                               method=self.selection_method, best_rate=self.best_rate)
        new = self._crossover(best, crossover=self.crossover,
                              n_parents=self.n_parents, method=self.crossover_method)
        self.population = new
        self._mutate(noise_generator, self.sigma)
        # best = torch.sort(evaluations)[0][0]
        return evaluations.min().item(), evaluations.max().item(), evaluations.mean().item()

    def _evaluate(self, X, Y):
        """
        Evaluate the population with respect to input data X and expected output.
        :param X: Input data, shape of (num batches, seq len, token length)
        :param Y: Expected output data.
        :return: Return an array (tensor) of evaluations, 1 for each network of the population.
        """
        # evaluate all networks
        #
        # evaluations = torch.zeros(self.population_size, device=self.device)
        evaluations = torch.zeros(self.population_size, device=self.device)

        for i in range(self.population_size):
            selected_pheno = self.population[i].cpu()
            # if IDCT is to be used first transform the vector, then use it to assemble the network
            if self.IDCT_from is not None:
                selected_pheno = torch.tensor(
                    fftpack.idct(np.array(selected_pheno), n=self.model.total_parameters(), norm="ortho"))
            fill_weights(self.model, selected_pheno.to(self.device))
            # evaluate
            predicted = self.model.forward(X)
            evaluations[i] = self.loss_function(predicted, Y)
        return evaluations

    def _mutate(self, noise_generator, sigma):
        """
        Performs the mutation step on newly generated individuals.
        :param noise_generator: A torch distribution, called as noise_generator.sample(*args)
        """

        mutation_indexes = torch.distributions.categorical.Categorical(
            torch.tensor([self.mutation_prob, 1 - self.mutation_prob])).sample([self.population_size]) > 0.5

        noise = noise_generator.sample([self.population_size, len(self.population[0])]).squeeze(-1)
        self.population[mutation_indexes] += noise[mutation_indexes] * sigma

    def _selection(self, evaluations, selection, method="truncated", best_rate=0.2):
        """
        given a list of evaluation scores, selects best_rate percentage of individuals by applying
        one of selection methods passed as parameter
        :param evaluations: fitness evaluations of the current population
        :param selection: apply or not selection
        :param method: selection method, possible methods are: "truncated", "fitness_based", "rank_based"
        :param best_rate: percentage of population to be selected
        :return: best population
        """

        if selection:
            end_range_for_parents = max(1, int(self.population_size * best_rate))
            evaluations_sorted = torch.sort(evaluations)
            population_sorted = self.population[evaluations_sorted[1]]

            if self.best_individual is None:
                self.best_individual = population_sorted[0]
                self.best_eval = evaluations_sorted[0][0]
            elif self.best_eval > evaluations_sorted[0][0]:
                self.best_individual = population_sorted[0]
                self.best_eval = evaluations_sorted[0][0]
            best_population = torch.zeros([end_range_for_parents, len(self.population[0])], device=self.device)
            if method == "truncated":
                """
                returns best individuals
                """
                best_population = population_sorted[:end_range_for_parents]
            elif method == "fitness_based":
                """
                probability of each individual to be selected is proportional to its fitness value
                """
                tot = sum(evaluations)
                probabilities = evaluations / tot
                for i in range(end_range_for_parents):
                    best_idx = torch.distributions.categorical.Categorical(
                        probabilities.clone().detach()).sample()
                    best_population[i] = self.population[best_idx]
                    # avoid repetitions
                    probabilities[best_idx] = 0
            elif method == "rank_based":
                """
                probability of each individual to be selected is proportional to its rank value
                """
                tot = ((1 + len(evaluations)) / 2) * len(evaluations)
                ranks = torch.linspace(1, len(evaluations), steps=len(evaluations), device=self.device)
                sorted_probabilities = 1 - ranks / tot
                for i in range(end_range_for_parents):
                    best_idx = torch.distributions.categorical.Categorical(
                        sorted_probabilities).sample()
                    best_population[i] = population_sorted[best_idx]
                    # avoid repetitions
                    sorted_probabilities[best_idx] = 0
            if self.elitism:
                best_population[end_range_for_parents - 1] = self.best_individual
        else:
            best_population = self.population
        return best_population

    def _crossover(self, best_population, crossover, n_parents=2, method="uniform_swap"):
        """
        given best population applies crossover with given probability, method and parents number
        :param best_population: vector of best individuals
        :param crossover: apply or not crossover
        :param n_parents: number of parents for single child creation
        :param method: crossover method to be applied, possible methods are: "single_swap", "uniform_swap", "arithmetic"
        :return: new population
        """
        if crossover:
            # randomly select parents
            parents_indexes = torch.randint(0, len(best_population), (self.population_size, n_parents),
                                            device=self.device)
            new_population = torch.zeros(self.population.shape, device=self.device)
            i = 0
            for p_idx in parents_indexes:
                new_population[i] = self._produce_child(best_population[p_idx], method=method)
                i += 1
        else:
            # randomly repeat best individuals
            new_pop_indexes = torch.randint(0, len(best_population), (self.population_size,), device=self.device)
            new_population = best_population[new_pop_indexes]
        return new_population

    def _produce_child(self, parents, method="uniform_swap"):
        """
        Given the list of parents and an index on where to place the new individual, creates a new individual
        :param parents: vector containing parents of new individual
        :param method: method to be applied, possible methods are: "single_swap", "uniform_swap", "arithmetic"
        """
        crossover_binary_op = None
        if method == "uniform_swap":
            crossover_binary_op = self._uniform_swap
        elif method == "single_swap":
            crossover_binary_op = self._single_swap
        elif method == "arithmetic":
            crossover_binary_op = lambda p1, p2: (p1 + p2) / 2
        child = parents[0].clone().detach()
        for parent in parents[1:]:
            child = crossover_binary_op(child, parent)
        return child

    def _uniform_swap(self, p1, p2):
        """
        performs uniform swap of p1 and p2: each element of p1 and p2 has the same probability to be selected
        :param p1: parent 1
        :param p2: parent 2
        :return: child
        """
        swap = torch.distributions.categorical.Categorical(
            torch.tensor([0.5, 0.5])).sample([len(p1)])
        p1_idx = (swap > 0.5)
        p2_idx = (swap < 0.5)
        child = torch.ones([len(p1)], device=self.device)
        child[p1_idx] = p1[p1_idx]
        child[p2_idx] = p2[p2_idx]
        return child

    def _single_swap(self, p1, p2):
        """
        performs single swap of p1 and p2: swap of random vector length is performed
        :param p1: parent 1
        :param p2: parent 2
        :return: child
        """
        swap_idx = torch.randint(0, len(p1), (1,), device=self.device)
        child = torch.ones(len(p1), device=self.device)
        child[:swap_idx] = p1[:swap_idx]
        child[swap_idx:] = p2[swap_idx:]
        return child

    def parameters(self):
        """
        Returns a dict mapping parameter names to values.
        :return:
        """
        res = dict()
        res["population_size"] = self.population_size
        res["mutation_prob"] = self.mutation_prob
        res["crossover"] = self.crossover
        res["selection"] = self.selection
        res["sigma"] = self.sigma
        res["crossover_method"] = self.crossover_method
        res["selection_method"] = self.selection_method
        res["best_rate"] = self.best_rate
        res["n_parents"] = self.n_parents
        res["model_parameters"] = self.model.total_parameters()
        res["IDCT_from"] = self.IDCT_from
        res["elitism"] = self.elitism
        return res

    @staticmethod
    def random_parameters():
        """
        Returns a dict of reasonable random parameters that can be used to do
        parameter search.
        :return: Dict of parameters for the algorithm.
        """
        res = dict()
        res["population_size"] = random.randrange(2, 21)
        res["mutation_prob"] = random.choice([0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
        res["crossover"] = random.choice([True, False])
        res["selection"] = random.choice([True, False])
        res["sigma"] = random.choice([0.1, 0.25, 0.5, 1])
        res["crossover_method"] = random.choice(["single_swap", "uniform_swap", "arithmetic"])
        res["selection_method"] = random.choice(["truncated", "fitness_based", "rank_based"])
        res["best_rate"] = random.choice([0.2, 0.3, 0.5])
        res["n_parents"] = random.choice([2, 3, 4])
        res["elitism"] = random.choice([True, False])
        return res

    @staticmethod
    def __name__():
        return "EA"
