import argparse
import torch
from code_useful.eval import output_useful
from transformers import BertTokenizer, BertModel
from code_useful.my_models import Useful_predict


def load_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    base_model = BertModel.from_pretrained("bert-base-uncased")
    self_model = Useful_predict(
        base_model,
        0.0,
        768,
        2
    ).double().to(device)
    self_model.load_state_dict(
        torch.load("../code_useful/outputs/1579143972/model_params.model")
    )
    
    return self_model


def analyze_useful(model, text):
    input_str = text.strip()
    predict_label, sentence_info = output_useful(model, input_str)
    return predict_label, sentence_info


if __name__ == '__main__':
    model = load_model()
    while True:
        input_str = input()
        returns_info = analyze_useful(model, input_str)
        print(returns_info)
