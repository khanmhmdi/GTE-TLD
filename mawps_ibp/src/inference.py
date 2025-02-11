"""
    The inference code.
    In this file, we will run the inference by using the prediction API \
        in the GeneratorInferenceWrapper.
    The GeneratorInferenceWrapper takes the raw inputs and produce the outputs.
"""
import random
import warnings
import numpy as np
import torch

from graph4nlp.pytorch.datasets.mawps import tokenize_mawps
from graph4nlp.pytorch.inference_wrapper.generator_inference_wrapper_for_tree import (
    GeneratorInferenceWrapper,
)
from graph4nlp.pytorch.models.graph2tree import Graph2Tree

from config import get_args

warnings.filterwarnings("ignore")


class Mawps:
    def __init__(self, opt=None):
        super(Mawps, self).__init__()
        self.opt = opt

        seed = self.opt["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.opt["gpuid"] == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:{}".format(self.opt["gpuid"]))
        self._build_model()

    def _build_model(self):
        self.model = Graph2Tree.load_checkpoint(self.opt["checkpoint_save_path"], "best.pt").to(
            self.device
        )

        self.inference_tool = GeneratorInferenceWrapper(
            cfg=self.opt, model=self.model, beam_size=2, lower_case=True, tokenizer=tokenize_mawps
        )

    @torch.no_grad()
    def translate(self):
        self.model.eval()
        ret = self.inference_tool.predict(
            raw_contents=[
                "add INT- 3 mul INT- 1 pow x INT+ 2"
            ],
            batch_size=5,
        )
        print('hi')
        print(ret)


if __name__ == "__main__":
    opt = get_args()
    runner = Mawps(opt)
    runner.translate()
