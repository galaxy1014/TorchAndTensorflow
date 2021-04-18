import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import random

# Train and test data
SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm')
LABEL = data.LabelField()

train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained = True)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(vars(train_data[-1]))

MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE,
                 vectors = 'glove.6B.100d', unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)

"""
* HUM for questions about humans
* ENTY for questions about entities
* DESC for questions asking you for a description
* NUM for questions where the answer is numerical
* LOC for questions where the answer is a location
* ABBR for questions asking about abbreviations

"""

print(LABEL.vocab.stoi)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE, device = device
)

# Create model

import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
               bidirectional, dropout, pad_idx):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
    self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                       bidirectional=bidirectional, dropout=dropout)
    self.fc = nn.Linear(hidden_dim * 2, output_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, text):
    embedded = self.dropout(self.embedding(text))
    # embedded = [sentence len, batch size, embedding dim]
    output, (hidden, cell) = self.rnn(embedded)

    # output = [sentence len, batch size, hidden dim * num directions]

    # hidden = [num layers * num directions, batch size, hidden dim]

    # cell = [num layers * num directions, batch size, hidden dim]

    hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

    # hidden = [batch size, hidden dim * num directions]

    return self.fc(hidden)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = len(LABEL.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
N_FILTERS = 100
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS,
            BIDIRECTIONAL, DROPOUT, PAD_IDX)

# Model setting

import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

def categorical_accuracy(preds, y):
  top_pred = preds.argmax(1, keepdim = True)
  correct = top_pred.eq(y.view_as(top_pred)).sum()
  acc = correct.float() / y.shape[0]
  return acc

def train(model, iterator, optimizer, criterion):
  epoch_loss = 0
  epoch_acc = 0

  model.train()

  for batch in iterator:
    optimizer.zero_grad()
    predictions = model(batch.text)
    loss = criterion(predictions, batch.label)
    acc = categorical_accuracy(predictions, batch.label)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
  epoch_loss = 0
  epoch_acc = 0

  model.eval()

  with torch.no_grad():
    for batch in iterator:
      predictions = model(batch.text)
      loss = criterion(predictions, batch.label)
      acc = categorical_accuracy(predictions, batch.label)
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

N_EPOCHS = 20

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
  start_time = time.time()

  train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
  valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

  end_time = time.time()

  epoch_mins, epoch_secs = epoch_time(start_time, end_time)

  if valid_loss < best_valid_loss:
    best_valid_loss = best_valid_loss
    torch.save(model.state_dict(), 'tut5-model.pt')

  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

# test performance

import spacy
nlp = spacy.load('en_core_web_sm')

def predict_class(model, sentence, min_len = 4):
  model.eval()
  tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
  if len(tokenized) < min_len:
    tokenized += ['<pad>'] * (min_len - len(tokenized))
  indexed = [TEXT.vocab.stoi[t] for t in tokenized]
  tensor = torch.LongTensor(indexed).to(device)
  tensor = tensor.unsqueeze(1)
  preds = model(tensor)
  max_preds = preds.argmax(dim = 1)
  return max_preds.item()

pred_class = predict_class(model, "Who is Keyser Söze?")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

pred_class = predict_class(model, "How many minutes are in six hundred and eighteen hours?")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

pred_class = predict_class(model, "What continent is Bulgaria in?")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

