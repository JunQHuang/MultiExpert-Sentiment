import torch
from torch import nn


class MultiExpertModel:

    def __init__(self):
        self.expert_models = self._init_models()

    def _init_models(self):
        return []


