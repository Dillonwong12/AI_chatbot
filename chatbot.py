import random
import json
import torch
import numpy as np
from model import NeuralNet
from utils import tokenize, lemmatize, bag_of_words

THRESHOLD = 0.6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
contexts = json.loads(open("contexts.json").read())
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
HIDDEN_SIZE = data["hidden_size"]
output_size = data["output_size"]
classes = data["classes"]
model_state = data["model_state"]
tfidf = data["tfidf"]

model = NeuralNet(input_size=input_size, hidden_size=HIDDEN_SIZE, num_classes=output_size).to(device)
model.load_state_dict(model_state)
model.eval()

print("Chatbot started. Type 'quit' to exit")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    X = tfidf.transform([" ".join([lemmatize(w) for w in tokenize(sentence)])])
    X = X.reshape(1, -1)
    X = torch.from_numpy(X.toarray()).to(device)

    output = model(X.to(torch.float32))

    _, predicted = torch.max(output, dim=1)
    tag = classes[predicted.item()]

    probs = torch.softmax(output, dim=1)
    # print(classes)
    # print([round(prob, 4) for prob in probs[0].tolist()])
    # print(type(probs))
    prob = probs[0][predicted.item()]
    

    if prob.item() > THRESHOLD:
        for context in contexts["contexts"]:
            if tag == context["tag"]:
                print(f"ChatBot: {random.choice(context['responses'])} ({round(prob.item(), 3)} confidence)")
    else:
        print("Sorry, I don't understand.")
