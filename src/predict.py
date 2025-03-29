import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the model and tokenizer
MODEL_PATH = "models/fake_news_bert.pth"  # Adjust path if different
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()  # Set model to evaluation mode

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = "Fake News" if torch.argmax(probs) == 1 else "Real News"
    confidence = probs.max().item() * 100
    return label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="News text to classify")
    args = parser.parse_args()

    label, confidence = predict(args.text)
    print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
