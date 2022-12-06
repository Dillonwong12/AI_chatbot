import random
import json
import torch
from model import NeuralNet
from utils import tokenize, bag_of_words

THRESHOLD = 0.75

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
intents = json.loads(open("intents.json").read())
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
HIDDEN_SIZE = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
classes = data["classes"]
model_state = data["model_state"]

model = NeuralNet(input_size=input_size, hidden_size=HIDDEN_SIZE, num_classes=output_size).to(device)
model.load_state_dict(model_state)
model.eval()

print("Chatbot started. Type 'quit' to exit")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, len(X))
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _, predicted = torch.max(output, dim=1)
    tag = classes[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > THRESHOLD:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"ChatBot: {random.choice(intent['responses'])}")
    else:
        print("Sorry, I don't understand.")
