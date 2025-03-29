import shap
import torch

def explain_with_shap(model, tokenizer, text):
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer([text])
    shap.plots.text(shap_values)
