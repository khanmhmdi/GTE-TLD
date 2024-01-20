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
        self.model = Graph2Tree.load_checkpoint(self.opt["checkpoint_save_path"], "epoch_345.pt").to(
            self.device
        )

        self.inference_tool = GeneratorInferenceWrapper(
            cfg=self.opt, model=self.model, beam_size=5, lower_case=True, tokenizer=tokenize_mawps
        )

    @torch.no_grad()
    def translate(self):
        self.model.eval()
        ret = self.inference_tool.predict(
            raw_contents=[
                "pow add pow x INT+ 2 mul INT+ 2 x INT+ 3"
            ],
            batch_size=1,
        )
        print('hi')
        print(ret)
# ['( ( ( - 3 ) * ( x ) ) + ( ( ( - 1 ) / ( 4 ) ) * ( ( x ) ^ ( 4 ) ) ) ) = 0']
# ['( ( ( - 3 ) * ( x ) ) + ( ( ( - 1 ) / ( 3 ) ) * ( ( x ) ^ ( 3 ) ) ) ) = 0']
# ( ( ( ( - 1 ) / ( 4 ) ) * ( ( x ) ^ ( 4 ) ) ) + ( ( ( ( 2 ) / ( 3 ) ) * ( ( x ) ^ ( 3 ) ) ) + ( ( ( - 1 ) + ( x ) ) * ( exp ( x ) ) ) ) ) = 0
# ['( ( ( ( 1 ) / ( 2 ) ) * ( ( x ) ^ ( 2 ) ) ) * ( cos ( ( 2 ) + ( x ) ) ) ) = 0']
# ['( ( ( -1 ) * ( ln ( cos ( x ) ) ) ) + ( ( -6 ) * ( ln ( cos ( x ) ) ) ) ) = 0']
#['( ( ( -1 ) * ( cos ( x ) ) ) + ( ( ( -6 ) * ( cos ( x ) ) ) + ( ( 3 ) * ( sin ( x ) ) ) ) ) = 0']

if __name__ == "__main__":
    opt = get_args()
    runner = Mawps(opt)
    runner.translate()
