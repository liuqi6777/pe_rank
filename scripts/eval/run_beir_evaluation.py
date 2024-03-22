import logging
import re
import torch
from torch import nn
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")


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


def mean_pooling(embeddings, attention_mask):
    return (torch.sum(embeddings * attention_mask.unsqueeze(-1), dim=1) \
        / torch.clamp(torch.sum(attention_mask, dim=1, keepdims=True), min=1e-9)).to(embeddings.dtype)


class EmbedderEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.embedder = AutoModel.from_pretrained(model_name, trust_remote_code=True).get_input_embeddings()
        self.embedder.cuda()

    @torch.no_grad()
    def encode(self, sentences, batch_size=512, **kwargs):
        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            inputs = self.tokenizer(sentences[i:i+batch_size], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            attention_mask = inputs["attention_mask"]
            embeddings.append(mean_pooling(self.embedder(inputs["input_ids"]), attention_mask).cpu())
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy()


# model_name = "jinaai/jina-embeddings-v2-base-en"
# model = ExtendedSentenceTransformer(
#     model_name,
#     "mlp2x_gelu",
#     "checkpoints/tiny2.jina.wiki1m.pretrain/projector.bin",
#     trust_remote_code=True,
# )
# model._first_module().max_seq_length = 512

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = EmbedderEncoder(model_name)


for task in TASK_LIST:
    logger.info(f"Running task: {task}")
    eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
    evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
    evaluation.run(
        model,
        output_folder=f"results/tiny",
        eval_splits=eval_splits,
        # overwrite_results=True,
)
