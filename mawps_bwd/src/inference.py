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

from graph4nlp.pytorch.datasets.mawps import tokenize_mawps, MawpsDatasetForTree
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
        from config import get_args

        para_dic = get_args()
        self.use_copy = self.opt["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.use_share_vocab = self.opt["graph_construction_args"]["graph_construction_share"][
            "share_vocab"
        ]
        self.data_dir = self.opt["graph_construction_args"]["graph_construction_share"]["root_dir"]

        para_dic = {
            "root_dir": self.opt["graph_construction_args"]["graph_construction_share"]["root_dir"],
            "word_emb_size": self.opt["graph_initialization_args"]["input_size"],
            "topology_subdir": self.opt["graph_construction_args"]["graph_construction_share"][
                "topology_subdir"
            ],
            "edge_strategy": self.opt["graph_construction_args"]["graph_construction_private"][
                "edge_strategy"
            ],
            "graph_name": self.opt["graph_construction_args"]["graph_construction_share"][
                "graph_name"
            ],
            "share_vocab": self.use_share_vocab,
            "enc_emb_size": self.opt["graph_initialization_args"]["input_size"],
            "dec_emb_size": self.opt["decoder_args"]["rnn_decoder_share"]["input_size"],
            "dynamic_init_graph_name": self.opt["graph_construction_args"][
                "graph_construction_private"
            ].get("dynamic_init_graph_name", None),
            "min_word_vocab_freq": self.opt["min_freq"],
            "pretrained_word_emb_name": self.opt["pretrained_word_emb_name"],
            "pretrained_word_emb_url": self.opt["pretrained_word_emb_url"],
            "pretrained_word_emb_cache_dir": self.opt["pretrained_word_emb_cache_dir"],
        }

        dataset = MawpsDatasetForTree(**para_dic)
        self.vocab_model = dataset.vocab_model
        if self.opt["gpuid"] == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:{}".format(self.opt["gpuid"]))
        self._build_model()

    def _build_model(self):

        self.model = Graph2Tree.from_args(self.opt, vocab_model=self.vocab_model)
        self.model.load_state_dict(torch.load(
            'C:/Users\Khanmhmdi\Desktop\graph4nlp-master\examples\pytorch\math_word_problem\mawps_bwd\src\examples\pytorch\math_word_problem\mawps_bwd\save\epoch_80.pt'))

        self.inference_tool = GeneratorInferenceWrapper(
                cfg=self.opt, model=self.model, beam_size=10, lower_case=True, tokenizer=tokenize_mawps
            )

    @torch.no_grad()
    def translate(self):
        self.model.eval()
        ret = self.inference_tool.predict(
            raw_contents=[
                " add INT+ 1 pow tan x INT+ 2"],
            batch_size=1,
        )
        print('hi')
        print(ret)


if __name__ == "__main__":
    opt = get_args()
    runner = Mawps(opt)
    runner.translate()
