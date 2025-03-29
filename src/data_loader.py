import pandas as pd
from datasets import load_dataset

def load_liar_dataset():
    dataset = load_dataset("liar")
    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])
    return train_df, val_df, test_df
