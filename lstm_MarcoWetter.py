"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
edited by Marco Wetter, last edited on 21st of July, 2019
"""
import numpy as np
from random import uniform
import sys
from math import tanh


# Since numpy doesn't have a function for sigmoid
# We implement it manually here
def sigmoid(x):
  return 1. / (1. + np.exp(-x))

'''
for the derivatives of the sigmoid and tanh function, it is actually good to add a 
small bias (e.g. 0.01) so that they do not saturate. in action this speeds up the training process, leads to smaller 
values of the loss function and better results
'''

# The derivative of the sigmoid function
def dsigmoid(y):
    return y * (1. - y) + 0.01

# The derivative of the tanh function
def dtanh(x):
    return 1. - x*x + 0.01


# The numerically stable softmax implementation
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# data I/O
data = open('data/input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
std = 0.1

option = sys.argv[1]

# hyperparameters
emb_size = 4
hidden_size = 32  # size of hidden layer of neurons
seq_length = 64  # number of steps to unroll the RNN for
learning_rate = 5e-2
max_updates = 250000

concat_size = emb_size + hidden_size

# model parameters
# char embedding parameters
Wex = np.random.randn(emb_size, vocab_size)*std # embedding layer

# LSTM parameters
Wf = np.random.randn(hidden_size, concat_size) * std # forget gate
Wi = np.random.randn(hidden_size, concat_size) * std # input gate
Wo = np.random.randn(hidden_size, concat_size) * std # output gate
Wc = np.random.randn(hidden_size, concat_size) * std # c term -> equals memory inside the LSTM Blocks

bf = np.zeros((hidden_size, 1)) # forget bias
bi = np.zeros((hidden_size, 1)) # input bias
bo = np.zeros((hidden_size, 1)) # output bias
bc = np.zeros((hidden_size, 1)) # memory bias

# Output layer parameters
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
by = np.zeros((vocab_size, 1)) # output bias


def forward(inputs, targets, memory):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """

    # The LSTM is different than the simple RNN that it has two memory cells
    # so here you need two different hidden layers
    hprev, cprev = memory

    hs, cs, xs, wes, zs, f_gate, i_gate, o_gate, c_hat, os, ys, ps = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    # hs:    hidden states of LSTM
    # cs:    characters; more precisely cs are the saved states insite the LSTM cell(s)
    # xs:    input to LSTM
    # wes:   word embeddings
    # zs:    stacked word embeddings and hidden states as suggested in the task description
    # f_gate:state of forget gate
    # i_gate:state of input gate
    # o_gate:state of output gate
    # c_hat: in between state of tanh(zs) + bc
    # os:    unnormalized log probabilities for next chars
    # ys:    output layers
    # ps:    probability distributions for the next chars

    # Here you should allocate some variables to store the activations during forward
    # One of them here is to store the hiddens and the cells
    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)

    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1

        # convert word indices to word embeddings
        wes[t] = np.dot(Wex, xs[t])

        # LSTM cell operation
        # first concatenate the input and h
        # This step is irregular (to save the amount of matrix multiplication we have to do)
        # zs is the concatenation of the input last hidden state with the word embeddings. 
        zs[t] = np.row_stack((hs[t-1], wes[t]))
        # YOUR IMPLEMENTATION should begin from here

        # Straight forward calculation of the gates, c_hat, hidden states etc.; 
        # Based on given code, the lecture slides of lecture 14: Recurrent neural networks
        # and the assumption, that the outputs os and ps are calculated in the same way as in the elman-rnn.py example,
        # meaning that the loss function is the cross entropy, and that p = softmax(o)
        # note that the candidate memory is names c_hat

        # compute the forget gate
        f_gate[t] = sigmoid(np.dot(Wf, zs[t]) + bf) # f_gate = sigmoid (Wf \cdot [h X] + bf)

        # compute the input gate
        i_gate[t] = sigmoid(np.dot(Wi, zs[t]) + bi) # i_gate = sigmoid (Wi \cdot [h X] + bi)

        # compute the candidate memory
        c_hat[t] = np.tanh(np.dot(Wc, zs[t]) + bc) # \hat{c} = tanh (Wc \cdot [h X] + bc])

        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        cs[t] = f_gate[t] * cs[t-1] + i_gate[t] * c_hat[t] # c_new = f_gate * prev_c + i_gate * \hat{c}

        # output gate
        o_gate[t] = sigmoid(np.dot(Wo, zs[t]) + bo) # o_gate = sigmoid (Wo \cdot [h X] + bo)

        # new hidden state for the LSTM
        hs[t] = o_gate[t] * np.tanh(cs[t]) # h = o_gate * tanh(c_new)

        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars

        os[t] = np.dot(Why, hs[t]) + by # o = Why \cdot h + by

        # softmax for probabilities for next chars
        ps[t] = softmax(os[t]) # p = softmax(o)

        # create an one hot vector for the label y
        ys[t] = np.zeros((vocab_size, 1))
        ys[t][targets[t]] = 1

        # and then cross-entropy (see the elman-rnn file for the hint)
        loss_t = np.sum(-np.log(ps[t]) * ys[t])
        loss += loss_t
    # define your activations
    memory = (hs[len(inputs)-1], cs[len(inputs)-1])
    # making sure that the activations are in the same order as we load in the backward function!
    activations = (hs, cs, xs, wes, zs, f_gate, i_gate, o_gate, c_hat, os, ys, ps)

    return loss, activations, memory


def backward(activations, clipping=True):

    # backward pass: compute gradients going backwards
    # Here we allocate memory for the gradients
    dWex, dWhy = np.zeros_like(Wex), np.zeros_like(Why)
    dby = np.zeros_like(by)
    dWf, dWi, dWc, dWo = np.zeros_like(Wf), np.zeros_like(Wi),np.zeros_like(Wc), np.zeros_like(Wo)
    dbf, dbi, dbc, dbo = np.zeros_like(bf), np.zeros_like(bi),np.zeros_like(bc), np.zeros_like(bo)

    # similar to the hidden states in the vanilla RNN
    # We need to initialize the gradients for these variables
    
    hs, cs, xs, wes, zs, f_gate, i_gate, o_gate, c_hat, os, ys, ps = activations
    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])

    # back propagation through time starts here
    for t in reversed(range(len(inputs))):
        # computing dL/do in the same way as in the example file elman-rnn.py
        # basically assuming that dL/do = p - y if p=softmax(o) and that the loss function is the cross entropy
        
        do = ps[t] - ys[t]

        # continuing with the next weights and biases used to create do. 
        # the chain rule does not yield any modifications to this with respect to the elman-rnn
        dWhy += np.dot(do, hs[t].T)
        dby += do

        # calculating dh. as it is connected to the old h and o we need to add the gradients
        dh = np.dot(Why.T, do) + dhnext

        '''
        it is noteworthy to explain additional terms in the backpropagation of the net here.
        in many cases the chain rule for a weight yields terms like
        dE/dWo = dE/dh * dh/do_gate * do_gate/dWo = dh * tanh(cs) * dsigmoid(o_gate) * zs; with o_gate = sigmoid(Wo * zs + bo)
        this is just the application of the chain rule, which results in the derivative of the sigmoid function(=o_gate)
        AND since we are deriving w.r.t. Wo also in a factor of zs. This factor needs to be transposed
        according to general rules of gradient descent with vectors and matrices. 
        in most cases this gives an additional factor of zs, which in itself ensures the right dimensions for the matrix gradients
        '''

        # because C it is connected to C we need to add the gradient of the c
        dC = dtanh(cs[t]) * o_gate[t] * dh + dcnext
        
        # now calculate the gradients on all the gates, and the weights and biases therein
        
        dWo += np.dot(dh*np.tanh(cs[t]) * dsigmoid(o_gate[t]),zs[t].T)
        dbo += dh * np.tanh(cs[t]) * dsigmoid(o_gate[t]) #os[t]*(1-os[t])
        
        # continuing with the calculation of c_hat and the input gate
        dWc += np.dot(dC * i_gate[t] * dtanh(c_hat[t]), zs[t].T)
        dbc += dC * i_gate[t] * dtanh(c_hat[t]) # dC_hat_pa
        
        dWi += np.dot(dC * c_hat[t] * dsigmoid(i_gate[t]), zs[t].T)
        dbi += dC * c_hat[t] * dsigmoid(i_gate[t]) # di_gate_pa

        # forget gate calculation
        dWf += np.dot(dC * cs[t-1] * dsigmoid(f_gate[t]), zs[t].T)
        dbf += dC * cs[t-1] * dsigmoid(f_gate[t]) # df_gate_pa
        
        # calculating the gradient on zs from all the gate gradients and then summing the errors up
        zf = np.dot(Wf.T, dC * cs[t-1] * dsigmoid(f_gate[t]))
        zi = np.dot(Wi.T, dC * c_hat[t] * dsigmoid(i_gate[t]))
        zc = np.dot(Wc.T, dC * i_gate[t] * dtanh(c_hat[t]))
        zo = np.dot(Wo.T, dh * np.tanh(cs[t]) * dsigmoid(o_gate[t]))
        dzs = zf + zi + zc + zo 

        # calculating the gradients on Wes, Wex, xs
        # since we just stacked hs[t-1] and the word embeddings, take the last [embedding_size] values of dzs as the error on the word embeddings,
        # and the first [hidden_size] values as the error on hs[t-1]
        dWes = dzs[hidden_size:,:]
        dWex += np.dot(dWes, xs[t].T)
        dxs = np.dot(Wex.T, dWes)

        # now making sure that dhnext and dcnext have the correct values for the next iteration
        # applying the forget gate to dcnext, which results from the derivative dE/dC[t-1] and C[t] = f_gate[t] * C[t-1] + i_gate[t]*c_hat[t]
        dhnext = dzs[:hidden_size,:]
        dcnext = dC * f_gate[t] 

    if clipping:
        # clip to mitigate exploding gradients
        for dparam in [dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby]:
            np.clip(dparam, -5, 5, out=dparam)

    gradients = (dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby)

    return gradients


def sample(memory, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    h, c = memory
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    generated_chars = []

    for t in range(n):
        # for this task just copied the calculation from the forward function, and omitted the step in which the data are saved
        we = np.dot(Wex, x)
        z = np.row_stack((h, we))
        f_gate = sigmoid(np.dot(Wf, z) + bf) # f_gate = sigmoid (Wf \cdot [h X] + bf)
        i_gate = sigmoid(np.dot(Wi, z) + bi) # i_gate = sigmoid (Wi \cdot [h X] + bi)
        c_hat = np.tanh(np.dot(Wc, z) + bc) # \hat{c} = tanh (Wc \cdot [h X] + bc])
        c = f_gate * c + i_gate * c_hat # c_new = f_gate * prev_c + i_gate * \hat{c}
        o_gate = sigmoid(np.dot(Wo, z) + bo) # o_gate = sigmoid (Wo \cdot [h X] + bo)
        h = o_gate * np.tanh(c) # h = o_gate * tanh(c_new)
        o = np.dot(Why, h) + by # o = Why \cdot h + by
        p = softmax(o) # p = softmax(o)

        # for the distribution, we randomly generate samples:
        ix = np.random.multinomial(1, p.ravel())
        x = np.zeros((vocab_size, 1))

        for j in range(len(ix)):
            if ix[j] == 1:
                index = j
        x[index] = 1
        generated_chars.append(index)

    return generated_chars


    return

if option == 'train':

    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(Wex), np.zeros_like(Why)
    mby = np.zeros_like(by) 

    mWf, mWi, mWo, mWc = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wo), np.zeros_like(Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bo), np.zeros_like(bc)

    smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
    
    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p+seq_length+1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size,1)) # reset RNN memory
            cprev = np.zeros((hidden_size,1))
            p = 0 # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

        # sample from the model now and then
        if n % 1000 == 0:
            sample_ix = sample((hprev, cprev), inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print ('----\n %s \n----' % (txt, ))

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = forward(inputs, targets, (hprev, cprev))
        gradients = backward(activations)

        hprev, cprev = memory
        dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients
        smooth_loss = smooth_loss * 0.99 + loss * 0.01
        if n % 1000 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                    [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                    [mWf, mWi, mWo, mWc, mbf, mbi, mbo, mbc, mWex, mWhy, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

        p += seq_length # move data pointer
        n += 1 # iteration counter
        n_updates += 1
        if n_updates >= max_updates:
            break

elif option == 'gradcheck':

    p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    delta = 0.001

    hprev = np.zeros((hidden_size, 1))
    cprev = np.zeros((hidden_size, 1))

    memory = (hprev, cprev)

    loss, activations, _ = forward(inputs, targets, memory)
    gradients = backward(activations, clipping=False)
    dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients

    for weight, grad, name in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by], 
                                   [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex    , dWhy, dby],
                                   ['Wf', 'Wi', 'Wo', 'Wc', 'bf', 'bi', 'bo', 'bc', 'Wex', 'Why', 'by']):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert(weight.shape == grad.shape), str_

        print(name)
        countidx = 0
        gradnumsum = 0
        gradanasum = 0
        relerrorsum = 0
        erroridx = []
        for i in range(weight.size):
            
            # evaluate cost at [x + delta] and [x - delta]
            w = weight.flat[i]
            weight.flat[i] = w + delta
            loss_positive, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w - delta
            loss_negative, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w  # reset old value for this parameter

            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / ( 2. * delta )
            gradnumsum += grad_numerical
            gradanasum += grad_analytic
            # compare the relative error between analytical and numerical gradients
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            if rel_error is None:
                rel_error = 0.
            relerrorsum += rel_error

            if rel_error > 0.01:
                #print ('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
                countidx += 1
                erroridx.append(i)
        print('For %s found %i bad gradients; with %i total parameters in the vector/matrix!' %(name, countidx, weight.size))
        print(' Average numerical grad: %0.9f \n Average analytical grad: %0.9f \n Average relative grad: %0.9f' %(gradnumsum/float(weight.size), gradanasum/float(weight.size), relerrorsum/float(weight.size)))
        print(' Indizes at which analytical gradient does not match numerical:', erroridx)
