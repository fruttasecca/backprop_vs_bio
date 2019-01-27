import torch
import datetime
import os
import pandas as pd

"""
Contains utility stuff such as logging results, parsing result files, etc.
"""


def is_float(value):
    try:
        float(value)
        return True
    except:
        return False


def is_integer(value):
    try:
        int(value)
        return True
    except:
        return False


def log_results(algorithm, task, run_times, min_loss_over_time, max_loss_over_time, avg_loss_over_time,
                seq_len_over_time):
    """
    Logs the results of a run.
    Runtimes, losses_over_time, and seq_len_over_time should have the same length.
    :param algorithm: Algorithm used.
    :param task: Task used (see tasks module).
    :param run_times: List of run times.
    :param min_loss_over_time: List of losses (min loss of that train step).
    :param max_loss_over_time: List of losses (max loss of that train step).
    :param avg_loss_over_time: List of losses (avg loss of that train step).
    :param seq_len_over_time: List of sequence lengths.
    :return:
    """
    # write results to directory
    # results/algorithm_name/task_name
    os.makedirs("results/%s/%s" % (algorithm.__class__.__name__, task.__str__), exist_ok=True)
    os.chdir("results/%s/%s" % (algorithm.__class__.__name__, task.__str__))
    filename = str(datetime.datetime.now()).replace(' ', '_')

    with open(str(filename), "w") as file:
        # write down algorithm and task parameters
        for k, v in algorithm.parameters().items():
            file.write("# %s %s\n" % (k, v))

        for k, v in task.parameters().items():
            file.write("# %s %s\n" % (k, v))

        # write down run time, loss and seq len as a csv table
        file.write("time,min_loss,max_loss,avg_loss,seq_len\n")
        for timing, min_loss, max_loss, avg_loss, seq_len in zip(run_times, min_loss_over_time, max_loss_over_time,
                                                                 avg_loss_over_time, seq_len_over_time):
            file.write("%s,%s,%s,%s,%s" % (timing, min_loss, max_loss, avg_loss, seq_len))
            file.write("\n")


def parse_result_file(filename):
    """
    Reads a result file, expected to be in the following format:

    # param paramvalue
    ...
    # param paramvalue
    time, loss, seq_len
    x,  x',  x''
    ...

    Which is essentialy a list of parameters followed by a csv table.
    Returns a pair made of (dictionary of parameters, pandas table containing the csv table).
    :param filename: Name of the file to parse.
    :return: (params dict, pandas table) pair.
    """
    # parse parameters
    parameters = dict()
    with open(filename, "r") as input:
        for line in input:
            if "#" == line[0]:
                _, k, v, = line.split()
                parameters[k] = v
            else:
                break
    # parse table
    data = pd.read_csv(filename, comment="#", sep=",")
    return parameters, data


def fill_weights(model, weights):
    """
    Given a torch nn model and an array of weights, replace the network weights
    with the ones from the passed array.
    :param model: Torch nn model.
    :param weights: Torch tensor.
    """
    start = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            required_weights = len(param.view(-1))
            # param.data = weights[start: start + required_weights].view(param.size())
            param.data = weights[start: start + required_weights].reshape(param.size()).clone()
            start += required_weights
    assert start == weights.size()[0]
    model.custom_flatten_parameters()


def network_to_array(model):
    """
    Given a torch nn model return an array (tensor of 1 dimension) that is the concatenation of all
    the learnable weights (required_grad) of the network.
    :param model: Torch nn model.
    """
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param.data.view(-1))
    model.custom_flatten_parameters()
    weights = torch.cat(params).contiguous()
    return weights


def mse_loss(predicted, expected):
    """
    MSE loss between 2 results.
    :param predicted:
    :param expected :
    :return:
    """
    mse = torch.nn.MSELoss()
    loss = mse(predicted, expected)
    return loss


def cross_loss(predicted, expected):
    """
    Cross entropy loss between predicted and expected output.
    :param predicted:
    :param expected:
    :return:
    """
    cross = torch.nn.CrossEntropyLoss()
    loss = cross(predicted.view(-1, predicted.size(2)), expected.view(-1))
    return loss


def parse_params_file(filename):
    """
    given a file containing a written dictionary of values, converts them to a dictionary
    :param filename:
    :return: dictionary
    """
    return_dict = eval(open(filename, 'r').read())
    for key in return_dict:
        if return_dict[key] == 'True':
            return_dict[key] = True
        elif return_dict[key] == 'False':
            return_dict[key] = False
        elif is_integer(return_dict[key]):
            return_dict[key] = int(return_dict[key])
        elif is_float(return_dict[key]):
            return_dict[key] = float(return_dict[key])
    return return_dict
