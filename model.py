import torch
import string
import numpy as np
import itertools
from collections import Counter
from tqdm import tqdm
import unidecode
from speechbrain.nnet.loss.transducer_loss import TransducerLoss


class Encoder(torch.nn.Module):
    def __init__(self, num_inputs):
        super(Encoder, self).__init__()
        self.embed = torch.nn.Embedding(num_inputs, encoder_dim)
        self.rnn = torch.nn.GRU(input_size=encoder_dim, hidden_size=encoder_dim, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1)
        self.linear = torch.nn.Linear(encoder_dim*2, joiner_dim)

    def forward(self, x):
        out = x
        out = self.embed(out)
        out = self.rnn(out)[0]
        out = self.linear(out)
        return out

class Predictor(torch.nn.Module):
    def __init__(self, num_outputs):
        super(Predictor, self).__init__()
        self.embed = torch.nn.Embedding(num_outputs, predictor_dim)
        self.rnn = torch.nn.GRUCell(input_size=predictor_dim, hidden_size=predictor_dim)
        self.linear = torch.nn.Linear(predictor_dim, joiner_dim)

        self.initial_state = torch.nn.Parameter(torch.randn(predictor_dim))
        self.start_symbol = NULL_INDEX # In the original paper, a vector of 0s is used; just using the null index instead is easier when using an Embedding layer.

    def forward_one_step(self, input, previous_state):
        embedding = self.embed(input)
        state = self.rnn.forward(embedding, previous_state)
        out = self.linear(state)
        return out, state

    def forward(self, y):
        batch_size = y.shape[0]
        U = y.shape[1]
        outs = []
        state = torch.stack([self.initial_state] * batch_size).to(y.device)
        for u in range(U+1): # need U+1 to get null output for final timestep 
            if u == 0:
                decoder_input = torch.tensor([self.start_symbol] * batch_size).to(y.device)
            else:
                decoder_input = y[:,u-1]
            out, state = self.forward_one_step(decoder_input, state)
            outs.append(out)
        out = torch.stack(outs, dim=1)
        return out


class Joiner(torch.nn.Module):
    def __init__(self, num_outputs):
        super(Joiner, self).__init__()
        self.linear = torch.nn.Linear(joiner_dim, num_outputs)

    def forward(self, encoder_out, predictor_out):
        out = encoder_out + predictor_out
        out = torch.nn.functional.relu(out)
        out = self.linear(out)
        return out

class Transducer(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Transducer, self).__init__()
        self.encoder = Encoder(num_inputs)
        self.predictor = Predictor(num_outputs)
        self.joiner = Joiner(num_outputs)

        if torch.cuda.is_available(): self.device = "cuda:0"
        else: self.device = "cpu"
        self.to(self.device)

    def compute_forward_prob(self, joiner_out, T, U, y):
        """
        joiner_out: tensor of shape (B, T_max, U_max+1, #labels)
        T: list of input lengths
        U: list of output lengths 
        y: label tensor (B, U_max+1)
        """
        B = joiner_out.shape[0]
        T_max = joiner_out.shape[1]
        U_max = joiner_out.shape[2] - 1
        log_alpha = torch.zeros(B, T_max, U_max+1).to(model.device)
        for t in range(T_max):
            for u in range(U_max+1):
                if u == 0:
                    if t == 0:
                        log_alpha[:, t, u] = 0.

                    else: #t > 0
                        log_alpha[:, t, u] = log_alpha[:, t-1, u] + joiner_out[:, t-1, 0, NULL_INDEX] 

                else: #u > 0
                    if t == 0:
                        log_alpha[:, t, u] = log_alpha[:, t,u-1] + torch.gather(joiner_out[:, t, u-1], dim=1, index=y[:,u-1].view(-1,1) ).reshape(-1)

                    else: #t > 0
                        log_alpha[:, t, u] = torch.logsumexp(torch.stack([
                          log_alpha[:, t-1, u] + joiner_out[:, t-1, u, NULL_INDEX],
                          log_alpha[:, t, u-1] + torch.gather(joiner_out[:, t, u-1], dim=1, index=y[:,u-1].view(-1,1) ).reshape(-1)
                        ]), dim=0)

        log_probs = []
        for b in range(B):
            log_prob = log_alpha[b, T[b]-1, U[b]] + joiner_out[b, T[b]-1, U[b], NULL_INDEX]
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs) 
        return log_prob

    def compute_loss(self, x, y, T, U):
        encoder_out = self.encoder.forward(x)
        predictor_out = self.predictor.forward(y)
        joiner_out = self.joiner.forward(encoder_out.unsqueeze(2), predictor_out.unsqueeze(1)).log_softmax(3)
        loss = -self.compute_forward_prob(joiner_out, T, U, y).mean()
        return loss

def compute_single_alignment_prob(self, encoder_out, predictor_out, T, U, z, y):
    """
    Computes the probability of one alignment, z.
    """
    t = 0; u = 0
    t_u_indices = []
    y_expanded = []
    for step in z:
        t_u_indices.append((t,u))
        if step == 0: # right (null)
            y_expanded.append(NULL_INDEX)
            t += 1
        if step == 1: # down (label)
            y_expanded.append(y[u])
            u += 1
    t_u_indices.append((T-1,U))
    y_expanded.append(NULL_INDEX)

    t_indices = [t for (t,u) in t_u_indices]
    u_indices = [u for (t,u) in t_u_indices]
    encoder_out_expanded = encoder_out[t_indices]
    predictor_out_expanded = predictor_out[u_indices]
    joiner_out = self.joiner.forward(encoder_out_expanded, predictor_out_expanded).log_softmax(1)
    logprob = -torch.nn.functional.nll_loss(input=joiner_out, target=torch.tensor(y_expanded).long().to(self.device), reduction="sum")
    return logprob



def compute_loss(self, x, y, T, U):
    encoder_out = self.encoder.forward(x)
    predictor_out = self.predictor.forward(y)
    joiner_out = self.joiner.forward(encoder_out.unsqueeze(2), predictor_out.unsqueeze(1)).log_softmax(3)
    #loss = -self.compute_forward_prob(joiner_out, T, U, y).mean()
    T = T.to(joiner_out.device)
    U = U.to(joiner_out.device)
    loss = transducer_loss(joiner_out, y, T, U) #, blank_index=NULL_INDEX, reduction="mean")
    return loss

