from scipy import fftpack
import torch
import random
import numpy as np

from util import fill_weights, network_to_array

"""
Differential evolution algorithm that can be used to train torch nn models. Uses binomial ("bin") crossover.
"""


class Differential(object):
    def __init__(self, device, model_class, model_args, loss_function, population_size, crossover_CR,
                 mutation_type, scale_factor, IDCT_from=None):
        """
        Creates an instance of the Differential Evolution algorithm.

        :param device: Torch device to be used.
        :param model_class:  The class of the torch network to use, so that it can be used as a constructor,
            called as model_class(**model_args).
        :param model_args: Dict containing the arguments to be used to call the constructor passed, as
            model_class(**model_args).
        :param crossover_CR: Value of the probability of crossover, range (0,1).
        :param mutation_type: Type of mutation to be used: "rand1", "rand2", "best1", "best2", "curtobest1", "randtobest2".
            See differential evolution or check their implementation/documentation to see how they work.
        :param scale_factor: Scale factor of the mutation.
        :param IDCT_from: If this value is an integer, the gene of each network will be represented by
            a vector of this length. Before using this gene to create a network an inverse discrete cosine
            transform will be applied to create the phenotype vector of the network, and from that the
            network will be "mounted"/assembled/created/ etc.
            This is a tentative to compress the network in order to reduce the search space.
            As seen in Evolving Neural Networks in Compressed Weight Space and others.
        """
        self.device = device
        self.loss_function = loss_function
        self.population_size = population_size
        self.crossover_CR = crossover_CR
        self.mutation_type = mutation_type
        self.scale_factor = scale_factor
        self.best_individual_index = 0
        self.best_individual_performance = 1e10
        self.individuals_perfomances = dict()
        self.model = model_class(**model_args)
        self.IDCT_from = IDCT_from

        assert 0. < crossover_CR < 1., "Crossover CR must be in range (0,1)"

        # check that mutation and crossover type are correct, and set them

        if self.mutation_type == "rand1":
            self._mutation = self._mutation_rand1
        elif self.mutation_type == "rand2":
            self._mutation = self._mutation_rand2
        elif self.mutation_type == "best1":
            self._mutation = self._mutation_best1
        elif self.mutation_type == "best2":
            self._mutation = self._mutation_best2
        elif self.mutation_type == "curtobest1":
            self._mutation = self._mutation_curtobest1
        elif self.mutation_type == "randtobest2":
            self._mutation = self._mutation_randtobest2
        else:
            print("Mutation type %s not recognized" % self.mutation_type)
            exit()

        # length of the genes
        self.total_gene_parameters = self.model.total_parameters() if self.IDCT_from is None else self.IDCT_from

        # init population, use a dict for faster indexing later (compared to a list)
        self.population = dict()
        for i in range(self.population_size):
            self.individuals_perfomances[i] = None

            genevector = network_to_array(model_class(**model_args))

            # transform using DCT if IDCT is being used
            if self.IDCT_from is not None:
                genevector = torch.tensor(
                    fftpack.dct(np.array(genevector.cpu()), n=self.total_gene_parameters, norm="ortho"))

            self.population[i] = genevector.to(self.device)

    def _crossover(self, gene1, gene2):
        """
        Performs a crossover between 2 genes, as in the crossover method in the differential evolution algorithm.
        Returns a new individual (genes), where at least 1 gene is guaranteed to be from gene2.
        :param gene1: First individual, gene as vector (torch tensor).
        :param gene1: Second individual, gene as vector (torch tensor).
        :return: A new individual, a vector of weights.
        """
        # gene that is for sure to be passed
        sure_pass = random.randrange(0, self.total_gene_parameters)

        # randomly select genes from parents
        # inherited from gene2 (mutant)
        mask2 = (torch.rand(self.total_gene_parameters, device=self.device) < self.crossover_CR).float()
        mask2[sure_pass] = 1.
        # inherited from gene1
        mask1 = 1. - mask2

        # create new gene
        newgene = mask2 * gene2 + mask1 * gene1
        return newgene

    """
    All different kinds of mutations, havent documented them all because they are private and the code
    is simple and self-explanatory.
    """

    def _mutation_rand1(self, gene):
        # get two unique individuals
        a, b = random.sample(range(self.population_size), 2)
        genea = self.population[a]
        geneb = self.population[b]

        # combine them
        mutant_gene = gene + self.scale_factor * (genea - geneb)
        return mutant_gene

    def _mutation_rand2(self, gene):
        # get unique individuals
        a, b, c, d = random.sample(range(self.population_size), 4)
        genea = self.population[a]
        geneb = self.population[b]
        genec = self.population[c]
        gened = self.population[d]

        # combine them
        mutant_gene = gene + self.scale_factor * (genea - geneb + genec - gened)
        return mutant_gene

    def _mutation_best1(self, gene):
        # get two unique individuals
        a, b = random.sample(range(self.population_size), 2)
        genebest = self.population[self.best_individual_index]
        genea = self.population[a]
        geneb = self.population[b]

        # combine them
        mutant_gene = genebest + self.scale_factor * (genea - geneb)
        return mutant_gene

    def _mutation_best2(self, gene):
        # get unique individuals
        a, b, c, d = random.sample(range(self.population_size), 4)
        genebest = self.population[self.best_individual_index]
        genea = self.population[a]
        geneb = self.population[b]
        genec = self.population[c]
        gened = self.population[d]

        # combine them
        mutant_gene = genebest + self.scale_factor * (genea - geneb + genec - gened)
        return mutant_gene

    def _mutation_curtobest1(self, gene):
        # get two unique individuals
        a, b = random.sample(range(self.population_size), 2)
        genebest = self.population[self.best_individual_index]
        genea = self.population[a]
        geneb = self.population[b]

        # combine them
        mutant_gene = gene + self.scale_factor * (genebest - gene + genea - geneb)
        return mutant_gene

    def _mutation_randtobest2(self, gene):
        # get unique individuals
        a, b, c, d = random.sample(range(self.population_size), 4)
        genebest = self.population[self.best_individual_index]
        genea = self.population[a]
        geneb = self.population[b]
        genec = self.population[c]
        gened = self.population[d]

        # combine them
        mutant_gene = genea + self.scale_factor * (genebest - gene + genea - geneb + genec - gened)
        return mutant_gene

    def parameters(self):
        """
        Returns a dict mapping parameter names to values.
        :return:
        """
        res = dict()
        res["population_size"] = self.population_size
        res["crossover_CR"] = self.crossover_CR
        res["mutation_type"] = self.mutation_type
        res["scale_factor"] = self.scale_factor
        return res

    def train_step(self, X, Y):
        """
        Given input data and expected output of the network, perform a train step of the cosyne algorithm on the
        batch of data, essentially moving on by 1 generation.
        The loss computation is carried out using the loss_function of this object instance.
        :param X: Input data, shape of (num batches, seq len, token length)
        :param Y: Expected output data.
        :return: Best, worst and average loss obtained in this generation.
        """
        for selected_index in range(self.population_size):
            selected_gene = self.population[selected_index]

            # check if individual has an evaluation, if it doesnt, evaluate it
            if self.individuals_perfomances[selected_index] is None:

                selected_pheno = selected_gene

                # if IDCT is to be used first transform the vector, then use it to assemble the network
                if self.IDCT_from is not None:
                    selected_pheno = torch.tensor(
                        fftpack.idct(np.array(selected_pheno.cpu()), n=self.model.total_parameters(), norm="ortho")).to(
                        self.device)

                # evaluate
                fill_weights(self.model, selected_pheno)
                predicted = self.model.forward(X)
                selected_ind_perf = self.loss_function(predicted, Y).item()
                self.individuals_perfomances[selected_index] = selected_ind_perf

                # update best performer if you must
                if selected_ind_perf < self.best_individual_performance:
                    self.best_individual_performance = selected_ind_perf
                    self.best_individual_index = selected_index

            # create mutant
            mutant_gene = self._mutation(selected_gene)

            # perform crossover
            offspring_genes = self._crossover(selected_gene, mutant_gene)

            offspring_pheno = offspring_genes

            # if IDCT is to be used first transform the vector, then use it to assemble the network
            if self.IDCT_from is not None:
                offspring_pheno = torch.tensor(
                    fftpack.idct(np.array(offspring_pheno.cpu()), n=self.model.total_parameters(), norm="ortho")).to(
                    self.device)

            # test performance of offspring
            fill_weights(self.model, offspring_pheno)
            predicted = self.model.forward(X)
            offspring_perf = self.loss_function(predicted, Y).item()
            if offspring_perf <= self.individuals_perfomances[selected_index]:
                # if offspring is better keep its weights, update performance
                self.individuals_perfomances[selected_index] = offspring_perf
                self.population[selected_index] = offspring_genes

                # update best performer if you must
                if offspring_perf < self.best_individual_performance:
                    self.best_individual_performance = offspring_perf
                    self.best_individual_index = selected_index

        # return performances
        performances = [perf for k, perf, in self.individuals_perfomances.items() if perf is not None]
        return np.min(performances), np.max(performances), np.mean(performances)

    def parameters(self):
        """
        Returns a dict mapping parameter names to values.
        :return:
        """
        res = dict()
        res["population_size"] = self.population_size
        res["crossover_CR"] = self.crossover_CR
        res["mutation_type"] = self.mutation_type
        res["scale_factor"] = self.scale_factor
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
        res["population_size"] = random.randrange(4, 21)
        res["crossover_CR"] = random.choice([0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
        res["mutation_type"] = random.choice(["rand1", "rand2", "best1", "best2", "curtobest1", "randtobest2"])
        res["scale_factor"] = random.choice([0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.80, 1., 1.20, 1.5, 1.75, 2.0])
        return res

    @staticmethod
    def __name__():
        return "differential"
