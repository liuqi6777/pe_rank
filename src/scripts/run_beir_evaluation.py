import argparse
import os
import re
import torch
from torch import nn
from mteb import MTEB
from sentence_transformers import SentenceTransformer


TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)


class Projector(nn.Sequential):
    def __init__(self, projector_type):
        super().__init__()
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))
        self.append(nn.Linear(768, 2048))
        for _ in range(1, mlp_depth):
            self.append(nn.GELU())
            self.append(nn.Linear(2048, 2048))

    def forward(self, x):
        return {"sentence_embedding": super().forward(x["sentence_embedding"])}


class ExtendedSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name, projector_type, projector_path, **kwargs):
        super().__init__(model_name, **kwargs)
        self.projector = Projector(projector_type)
        self.projector.load_state_dict(torch.load(projector_path, map_location="cpu"), strict=False)
        self.add_module("projector", self.projector)
    
    def encode(self, sentences, **kwargs):
        kwargs.pop("normalize_embeddings", None)
        return super().encode(sentences, normalize_embeddings=True, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="jinaai/jina-embeddings-v2-base-en")
    parser.add_argument("--projector-path", type=str, default=None)
    args = parser.parse_args()

    if not args.projector_path:
        model = SentenceTransformer(args.model_name, trust_remote_code=True)
        output_path = os.path.join("results", args.model_name.split("/")[-1])
    else:
        model = ExtendedSentenceTransformer(
            args.model_name,
            "mlp2x_gelu",
            args.projector_path,
            trust_remote_code=True,
        )
        output_path = os.path.join("results", "mteb", args.projector_path.split("/")[-1])
    model._first_module().max_seq_length = 512

    for task in TASK_LIST:
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        evaluation.run(
            model,
            output_folder=output_path,
            eval_splits=eval_splits,
    )
