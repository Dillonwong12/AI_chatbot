import json
import pandas as pd

IGNORE_CONTEXTS = ['appropriate disclosure', 'unknown']

data = pd.read_excel("D:/System Default/Desktop/NLU_2022/Emotion_Context_Datasets/emotion_context_wrangle/context_classification_eng_1.xlsx")

contexts = []
for context in list(data['context'].unique()):
    if context not in IGNORE_CONTEXTS:
        context_dict = {}
        patterns = (data['text'][data['context'] == context].values.tolist())
        patterns = [pattern.strip() for pattern in patterns]
        context_dict['tag'] = context
        context_dict['patterns'] = patterns
        context_dict['responses'] = [context + " detected."]
        contexts.append(context_dict)

with open("contexts.json", "w") as FILE:
    json.dump({"contexts": contexts}, FILE, indent=4)