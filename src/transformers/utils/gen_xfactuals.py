import torch
import torch.nn as nn
import torch.optim as optim


def gen_counterfactual(z, probe, s_prime, criterion=None):
    z_prime = z
    z_prime.requires_grad = True
    # optimizer = optim.SGD([z_prime], lr=0.0001, momentum=0.9)
    # optimizer = optim.SGD([z_prime], lr=0.001, momentum=0.9)
    optimizer = optim.SGD([z_prime], lr=0.01, momentum=0.9)  # Good. Generated the prey results.
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    num_steps = 0
    stopping_loss = 0.001  # Was 0.05
    # stopping_loss = .001  # Generated the prey results
    loss = 100
    max_patience = 10000
    max_num_steps = 100  # FIXME
    curr_patience = 0
    min_loss = loss
    probe.eval()
    while num_steps < max_num_steps and loss > stopping_loss:
        optimizer.zero_grad()
        outputs = probe(z_prime)
        loss = criterion(outputs, s_prime)
        loss.backward()
        optimizer.step()
        if num_steps % 100 == 0:
            print("Loss", loss)
        num_steps += 1
        curr_patience += 1
        if loss < min_loss - 0.01:
            min_loss = loss
            curr_patience = 0
        if curr_patience > max_patience:
            print("Breaking because of patience with loss", loss)
            break
    print("Num steps", num_steps, "\tloss", loss)
    return z_prime
