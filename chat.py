import random
import json
import torch
import logging
import time
from datetime import datetime 

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Add logging configuration
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f'session_{session_id}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

logging.info("==== New Chat Session Started ====")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('cricket_intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    start_time = time.time()
    
    if sentence == "quit":
        break

    # Log user input
    logging.info(f"User Input: {sentence}")
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    processing_time = time.time() - start_time
    
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}")
                logging.info(f"Bot Response: {response} (tag: {tag}, confidence: {prob.item():.4f})")
    else:
        print(f"{bot_name}: I do not understand...")
        logging.info(f"Bot Response: I do not understand... (confidence: {prob.item():.4f})")
    
    logging.info(f"Processing time: {processing_time:.4f} seconds")