import torch
import torch.nn.functional as F
import random
import time 

num_args = 27
vector_dim = 7
block_size = 4
num_neurons = 300
learning_rate = 0.1
mini_batches = 50

def read_dataset(): 
    global words, stoi, itos
    words = open(r'C:\Users\jsh27\Downloads\Jupyer Notebook\names.txt', 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    stoi

    itos = {i:s for s,i in stoi.items()}



def build_dataset(word_list):

    X = []
    Y = []

    for word in word_list:
        context = [0]*block_size
        for c in word + '.':
            ic = stoi[c]
            X.append(context)
            Y.append(ic)
            # print (f'{''.join(itos[i] for i in context)}---->{c}')
            context = context[1:] + [ic]
        
        # print("--------------------")

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y
   


def separate_sets():

    random.shuffle(words)
    
    n1 = int(0.8*(len(words)))
    n2 = int(0.9*(len(words)))

    Xtr, Ytr = build_dataset(words[0:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xtest, Ytest = build_dataset(words[n2:])

    return Xtr, Ytr, Xdev, Ydev, Xtest, Ytest

def initialize_parameters():
    # embedding matrix
    C = torch.randn((num_args,vector_dim))
    W1 = torch.randn(((block_size * vector_dim),num_neurons))
    B1 = torch.randn(num_neurons)
    W2 = torch.randn((num_neurons,num_args))
    B2 = torch.randn(num_args)

    parameters = [C,W1,B1,W2,B2]

    for p in parameters:
        p.requires_grad = True

    return parameters


def training(X,Y, parameters, steps = 1000):

    C, W1, B1, W2, B2 = parameters
    

    start = time.time()

    for i in range(steps):

        ix = torch.randint(0, X.shape[0],(mini_batches,))
        iy = Y[ix]

        # forward pass
        emb = C[X[ix]]
        h = torch.tanh((emb.view(emb.shape[0], block_size* vector_dim))@W1 + B1)
        logits = h@W2 + B2
        loss = F.cross_entropy(logits, iy)

        # backward pass
        for p in parameters:
            p.grad = None
        
        loss.backward()


        # update
        for p in parameters:
            p.data += -p.grad*learning_rate

        # print (loss.item())

        if i% 100 == 0:
            print(f'Time Elapsed -- {time.time() - start} | Loss: {loss.item()}')

    print(loss.item())

    return parameters

def testing(X,Y,parameters):

    C, W1, B1, W2, B2 = parameters

    emb = C[X]
    h = torch.tanh((emb.view(emb.shape[0], block_size* vector_dim))@W1 + B1)
    logits = h@W2 + B2
    loss = F.cross_entropy(logits, Y)

    print(loss.item())


if __name__ == "__main__":

    read_dataset()
    Xtr, Ytr, Xdev, Ydev, Xtest, Ytest = separate_sets()
    params = initialize_parameters()
    training(Xtr, Ytr, params, 8000)
    learning_rate = 0.01
    training(Xtr, Ytr, params, 1000)
    learning_rate = 0.001
    model_params = training(Xtr, Ytr, params, 1000)
    testing(Xtest, Ytest, model_params)






    







    
    




        
            




    



