import json
import torch
import time
import nltk
import random
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_bleu(reference_responses, generated_response):
    """Calculate BLEU score using all available reference responses"""
    reference_tokens = [word_tokenize(ref.lower()) for ref in reference_responses]
    generated_tokens = word_tokenize(generated_response.lower())
    return sentence_bleu(reference_tokens, generated_tokens)

def get_chatbot_response(model, device, all_words, tags, intents, input_text):
    """Get response from chatbot for given input"""
    sentence = tokenize(input_text)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        tag = tags[predicted.item()]
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses']), tag, prob.item()
    return "I do not understand...", None, prob.item()

def evaluate_model(test_cases):
    correct = 0
    total = 0
    confidence_scores = []
    bleu_scores = []
    total_time = 0

    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load("data.pth")
    model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()

    # Load intents
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    for test_input, expected_tag in test_cases:
        start_time = time.time()

        # Get chatbot's response
        bot_response, predicted_tag, confidence = get_chatbot_response(
            model, device, data['all_words'], data['tags'], intents, test_input
        )

        end_time = time.time()
        processing_time = end_time - start_time
        total_time += processing_time

        correct += (predicted_tag == expected_tag if predicted_tag else False)
        total += 1
        confidence_scores.append(confidence)

        # Get multiple reference responses for BLEU
        reference_responses = []
        for intent in intents['intents']:
            if intent['tag'] == expected_tag:
                reference_responses = intent['responses']
                break

        # Calculate BLEU score using multiple references
        if reference_responses and bot_response != "I do not understand...":
            bleu_score = calculate_bleu(reference_responses, bot_response)
            bleu_scores.append(bleu_score)

        # Print individual test case results
        print(f"\nTest Case {total}:")
        print(f"Input: {test_input}")
        print(f"Expected Tag: {expected_tag}")
        print(f"Predicted Tag: {predicted_tag}")
        print(f"Bot Response: {bot_response}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Processing Time: {processing_time:.4f} seconds")
        if reference_responses and bot_response != "I do not understand...":
            print(f"BLEU Score: {bleu_score:.4f}")
        print("---")

    # Print overall metrics
    print("\nEvaluation Metrics:")
    print(f"Overall Accuracy: {(correct/total)*100:.2f}%")
    print(f"Average Confidence: {sum(confidence_scores)/len(confidence_scores):.4f}")
    print(f"Average Processing Time: {(total_time/total)*1000:.4f} milliseconds")
    print(f"Total Evaluation Time: {total_time*1000:.4f} milliseconds")
    if bleu_scores:
        print(f"Average BLEU Score: {sum(bleu_scores)/len(bleu_scores):.4f}")

# Test cases
test_cases = [
    ("theme of Updates 2k24?", "theme"),
    ("tell me the title sponsors for Updates.", "title sponsors"),
    ("main faculty coordinators of Updates?", "faculty coordinators"),
    ("Names of members of web developement team", "web_developers"),
    ("Where can i get news of Updates event?", "social_media"),
    ("what is man in the middle event?", "Man_In_Middle"),
    ("What is Human or Ai event?", "Human_Or_Ai"),
]

if __name__ == "__main__":
    evaluate_model(test_cases)
