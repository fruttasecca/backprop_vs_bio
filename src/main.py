#!/usr/bin/python3
import torch
import time
import sys

# local imports
import lstm_autoenc
import autoenc
import tasks
import util
from util import mse_loss, cross_loss
import backprop
import cosyne
import differential
import ea


def run_task(algorithm, task):
    """
    Runs an algorithm to train a neural network on the given task.
    :param algorithm:
    :param task:
    :return:
    """
    # training
    # logging stuff
    runtimes = []
    min_loss_over_time = []
    max_loss_over_time = []
    avg_loss_over_time = []
    seq_len_over_time = []
    start_time = time.time()
    min_error = 1e10

    for X, Y in task:
        min_loss, max_loss, avg_loss = algorithm.train_step(X, Y)
        min_error = min(min_error, min_loss)

        # log data
        min_loss_over_time.append(min_loss)
        max_loss_over_time.append(max_loss)
        avg_loss_over_time.append(avg_loss)

        time_from_start = time.time() - start_time
        runtimes.append(time_from_start)
        seq_len_over_time.append(X.size()[1])
    print("elapsed %s seconds" % (time.time() - start_time))
    print("best result: %s" % min_error)

    # log results to a file named as the current timestamp, position (directory) depends on algorithm, task
    util.log_results(algorithm, task, runtimes, min_loss_over_time, max_loss_over_time, avg_loss_over_time,
                     seq_len_over_time)


"""
distr nn uniformi
bias?
"""

if __name__ == "__main__":
    args = sys.argv[1:]
    assert len(
        args) >= 8, "Expected at least 8 arguments:" \
                    " size, algorithm, task, numclasses, IDCT, batch_size, seq_len, total_batches," \
                    " [algorithm_params_filename]"
    size, algorithm, task, numclasses, IDCT, batch_size, seq_len, total_batches = args[:8]
    alg_parameters = None
    if len(args) == 9:
        alg_parameters_file = args[8]
        alg_parameters = util.parse_params_file(alg_parameters_file)
    size = int(size)
    numclasses = int(numclasses)
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    total_batches = int(total_batches)

    IDCT = None if IDCT == "None" else int(IDCT)

    assert size > 0
    assert size >= 4 or task != "enc"
    assert algorithm in ["backprop", "cosyne", "differential", "ea"]
    assert task in ["mse", "class", "enc"]
    assert numclasses > 0

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # neural network to train
    if task != "enc":
        model = lstm_autoenc.AutoEncoder
        model_args = dict()
        model_args["device"] = device
        model_args["input_dim"] = 1 if task == "mse" else numclasses
        model_args["hidden_dim"] = size
    else:
        model = autoenc.AutoEncoder
        model_args = dict()
        model_args["device"] = device
        model_args["input_dim"] = size

    # algorithm to train it
    loss_function = mse_loss if (task == "mse" or task == "enc") else cross_loss

    if algorithm == "backprop":
        if alg_parameters is None:
            algorithm = backprop.Backprop(device=device, model_class=model, model_args=model_args,
                                          loss_function=loss_function, **backprop.Backprop.random_parameters()).to(
                device)
        else:
            algorithm = backprop.Backprop(device=device, model_class=model, model_args=model_args,
                                          loss_function=loss_function, **alg_parameters).to(
                device)
    elif algorithm == "cosyne":
        if alg_parameters is None:
            algorithm = cosyne.Cosyne(device=device, model_class=model, model_args=model_args,
                                      loss_function=loss_function, **cosyne.Cosyne.random_parameters(), IDCT_from=IDCT)
        else:
            algorithm = cosyne.Cosyne(device=device, model_class=model, model_args=model_args,
                                      loss_function=loss_function, **alg_parameters, IDCT_from=IDCT)

        # algorithm = cosyne.Cosyne(device, autoenc_class, model_args, loss_function, 4, 0.2, 2)# IDCT_from=20)
    elif algorithm == "differential":
        if alg_parameters is None:
            algorithm = differential.Differential(device=device, model_class=model, model_args=model_args,
                                                  loss_function=loss_function,
                                                  **differential.Differential.random_parameters(), IDCT_from=IDCT)
        else:
            algorithm = differential.Differential(device=device, model_class=model, model_args=model_args,
                                                  loss_function=loss_function,
                                                  **alg_parameters, IDCT_from=IDCT)
    elif algorithm == "ea":
        if alg_parameters is None:
            algorithm = ea.EA(device=device, model_class=model, model_args=model_args,
                              loss_function=loss_function,
                              **ea.EA.random_parameters(), IDCT_from=IDCT)
        else:
            algorithm = ea.EA(device=device, model_class=model, model_args=model_args,
                              loss_function=loss_function,
                              **alg_parameters, IDCT_from=IDCT)
        # algorithm = differential.Differential(device=device, model_class=autoenc_class, model_args=model_args,
        #                                       loss_function=loss_function, population_size=4, crossover_CR=0.2,
        #                                       mutation_type="curtobest1", scale_factor=1.5, IDCT_from=20)

    # task to perform
    if task == "mse":
        task = tasks.intDataLoader(device, batch_size=batch_size, seq_len=seq_len, min=0, max=numclasses,
                                   total_batches=total_batches,
                                   linear_inc=None,
                                   seed=1337)
    elif task == "enc":
        task = tasks.realValuedDataLoader(device, batch_size=batch_size, dimension=size, total_batches=total_batches,
                                          seed=1337)
    else:
        task = tasks.classDataLoader(device, batch_size=batch_size, seq_len=seq_len, num_classes=numclasses,
                                     total_batches=total_batches,
                                     linear_inc=None,
                                     seed=1337)

    print("running parameters")
    print(algorithm.parameters())
    run_task(algorithm, task)
