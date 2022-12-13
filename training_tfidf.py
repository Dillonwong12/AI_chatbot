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
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000, max_df=0.80, ngram_range=(1, 2), stop_words=stopwords.words('english'))

# model output
FILE = "data.pth"

# hyperparams
BATCH_SIZE = 8
HIDDEN_SIZE = 8
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000

contexts = json.loads(open("contexts.json").read())
X = []
y = []
classes = []
xy = []

for context in contexts['contexts']:
    for pattern in context['patterns']:
        pattern = " ".join([lemmatize(w) for w in tokenize(pattern) if not re.match(r'[^\w\s]', w) 
            and w not in stopwords.words('english') and w != '' and not any(char.isdigit() for char in w)])
        if len(pattern) > 1:
            X.append(pattern)
            y.append(context['tag'])

classes = sorted(set(y))
print(X[:5])
X = tfidf.fit_transform(X).toarray()

print(classes)

for idx, tag in enumerate(y):
    label = classes.index(tag)
    y[idx] = label

X_train = np.array(X)
y_train = np.array(y)

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
        words = words.to(device).to(torch.float32)
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
    "classes": classes,
    "tfidf": tfidf
}

torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')