from llmai.model import ModelPretrainForLLM
from llmai.dataset import DatasetForLLM


datasets = {
   "llm": DatasetForLLM
}

models = {
    "mistral_fintune": ModelPretrainForLLM,
}