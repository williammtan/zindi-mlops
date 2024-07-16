"""Kserve inference script."""

import argparse
from kserve import (
    Model,
    ModelServer,
    model_server,
    InferRequest,
    InferOutput,
    InferResponse,
)
from kserve.utils.utils import generate_uuid
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

HF_MODEL_NAME = "./saved_model/"

class MachineTranslation(Model):
    """Kserve inference implementation of model."""

    def __init__(self, name: str):
        """Initialise model."""
        super().__init__(name)
        self.model = None
        self.tokenizer = None
        self.ready = False
        self.load()

    def load(self):
        """Reconstitute model from huggingface."""
        print("Loading models...")
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
        print("Models loaded!")
        self.ready = True

    def predict(self, payload: InferRequest, *_args, **_kwargs) -> InferResponse:
        """Pass inference request to model to make prediction."""
        print("Starting predict..")
        print(f"{payload.inputs[0].data[0]}")

        response_id = generate_uuid()
        inputs = self.tokenizer(
            payload.inputs[0].data[0],
            return_tensors="pt",
            max_length=64,
            truncation=True,
            padding=True
        )["input_ids"]
        outputs = self.model.generate(
            inputs,
            max_length=64,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("fra_Latn")
        )
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        infer_output = InferOutput(
            name="PREDICTION__0", datatype="BYTES", shape=[1], data=[translated_text]
        )
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        return infer_response


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument("--protocol", help="The protocol for the predictor", default="v2")
parser.add_argument("--model_name", help="The name that the model is served under.")
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MachineTranslation(args.model_name)
    print("Starting Kserve server...")
    ModelServer().start([model])
