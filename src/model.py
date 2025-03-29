from transformers import BertForSequenceClassification, BertTokenizer
import torch

MODEL_NAME = "bert-base-uncased"

def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    return model, tokenizer
