from torchtext import data
from torchtext import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm

import models

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    
    #round predictions to the closest integer after squashing them with the sigmoid function
    rounded_preds = torch.round(F.sigmoid(preds)) 

    #find how many predictions match targets and convert into float for division 
    correct = (rounded_preds == y).float() 

    #accuracy is number correct / batch size
    acc = correct.sum()/len(correct) 

    return acc

BATCH_SIZE = 32
HIDDEN_DIM = 10
N_EPOCHS = 5
N_GRAMS = 1
VOCAB_MAX_SIZE = None
VOCAB_MIN_FREQ = 1
MAX_LENGTH = None
TOKENIZER = lambda s: s.split() 

def generate_n_grams(x):
    """
    Takes in a list of strings.
    Generates n-grams from that list and appends to the end of the list.
    """

    if N_GRAMS <= 1: #no need to create n-grams if we only want uni-grams
        return x
    else:
        n_grams = set(zip(*[x[i:] for i in range(N_GRAMS)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x

# set up fields
TEXT = data.Field(batch_first=True, tokenize=TOKENIZER, preprocessing=generate_n_grams, fix_length=MAX_LENGTH)
LABEL = data.Field(batch_first=True, pad_token=None, unk_token=None)

# make splits for data
train, test = datasets.IMDB.splits(TEXT, LABEL)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

# build the vocabulary
TEXT.build_vocab(train, max_size=VOCAB_MAX_SIZE, min_freq=VOCAB_MIN_FREQ)
LABEL.build_vocab(train)

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))

# make iterator for splits
train_iter, test_iter = data.BucketIterator.splits(
    (train, test), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.text),
    device = None if torch.cuda.is_available() else -1, #device needs to be -1 for CPU, else use default GPU
    repeat=False
    )

#initialize model
model = models.FastText(len(TEXT.vocab), HIDDEN_DIM, 1)

#initialize optimizer, scheduler and loss function
optimizer = optim.Adam(model.parameters())
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
criterion = nn.BCEWithLogitsLoss()

#place on GPU
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

for epoch in range(1, N_EPOCHS+1):

    #set metric accumulators
    epoch_loss = 0
    epoch_acc = 0

    for batch in tqdm(train_iter, desc='Train'):

        #get inputs and targets from batch
        x = batch.text
        y = batch.label.float() #this is a long by default

        #zero the gradients from last backprop
        optimizer.zero_grad()

        #pass inputs forward and get predictions
        predictions = model(x)

        #calculate loss/cost from predictions and targets
        loss = criterion(predictions, y)

        #backpropagate to calculate the gradients
        loss.backward() 

        #apply the gradients
        optimizer.step()

        #calculate accuracy
        acc = binary_accuracy(predictions, y)

        #accumulate metrics across epoch
        epoch_loss += loss.data[0]
        epoch_acc += acc.data[0]

    #calculate metrics averaged across whole epoch
    train_acc = epoch_acc / len(train_iter)
    train_loss = epoch_loss / len(train_iter)

    #reset metric accumulators
    epoch_loss = 0
    epoch_acc = 0 

    for batch in tqdm(test_iter, desc=' Test'):

        #get inputs and targets from batch
        x = batch.text
        y = batch.label.float() #this is a long by default

        #pass inputs forward and get predictions
        predictions = model(x)

        #calculate loss/cost from predictions and targets
        loss = criterion(predictions, y)

        #calculate accuracy
        acc = binary_accuracy(predictions, y)

        #accumulate metrics across epoch
        epoch_loss += loss.data[0]
        epoch_acc += acc.data[0]

    #calculate metrics averaged across whole epoch
    test_acc = epoch_acc / len(test_iter)
    test_loss = epoch_loss / len(test_iter)

    #update the scheduler
    scheduler.step(test_loss)

    #print metrics
    print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Acc.: {train_acc*100:.2f}%, Test Loss: {test_loss:.3f}, Test Acc.: {test_acc*100:.2f}%')

