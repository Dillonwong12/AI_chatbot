import random
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import tokenize, stem, lemmatize, bag_of_words
from model import NeuralNet

# model output
FILE = "model_data.pth"

# hyperparams
BATCH_SIZE = 8
HIDDEN_SIZE = 8
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000

contexts = json.loads(open("contexts.json").read())
words = []
classes = []
xy = []

for context in contexts['contexts']:
    for pattern in context['patterns']:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        xy.append((word_list, context['tag']))
        classes.append(context['tag'])

words = [lemmatize(w) for w in words if not re.match(r'[^\w\s]', w) and w not in stopwords.words('english')]
all_words = sorted(set(words))
classes = sorted(set(classes))

print(classes)
print(all_words)

X_train = []
y_train = []

for pattern_sentence, tag in xy:
    bow = bag_of_words(pattern_sentence, all_words)
    X_train.append(bow)

    label = classes.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


train_loader = DataLoader(dataset=ChatDataset(), batch_size=BATCH_SIZE, shuffle=True)
input_size = len(X_train[0])
output_size = len(classes)

print(input_size, output_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size=input_size, hidden_size=HIDDEN_SIZE, num_classes=output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device).type(torch.LongTensor)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{NUM_EPOCHS}, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": HIDDEN_SIZE,
    "all_words": all_words,
    "classes": classes
}
print(data['all_words'], len(data["all_words"]))

torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')