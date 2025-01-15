from llmai.model import ModelPretrainForLLMFintune
from llmai.dataset import DatasetForLLM


datasets = {
   "llm": DatasetForLLM
}

models = {
    "mistral": ModelPretrainForLLMFintune,
}