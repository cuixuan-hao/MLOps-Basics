import torch
from model import ColaModel
from data import DataModule

class ColaPredictor:
    def __init__(self, model_path):
        self.model = ColaModel.load_from_checkpoint(model_path).eval().freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = ["unacceptable", "acceptable"]

    def predict(self, sentence: str):
        tokens = self.processor.tokenize_data({"sentence": sentence})
        input_ids = torch.tensor([tokens["input_ids"]])
        attention_mask = torch.tensor([tokens["attention_mask"]])
        logits = self.model(input_ids, attention_mask)[0]
        probs = self.softmax(logits).tolist()
        return [{"label": label, "score": prob} for label, prob in zip(self.labels, probs)]

if __name__ == "__main__":
    predictor = ColaPredictor("./models/epoch=0-step=267.ckpt")
    result = predictor.predict("The boy is sitting on a bench")
    print(result)
