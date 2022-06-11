import json

from .types_ import *


def get_data(path: str) -> List[Tuple]:
    with open(path, "r", encoding="utf-8") as f:
        jsonl = list(f)

    datasets = []
    for json_str in jsonl:
        datasets.append(json.loads(json_str))

    data = []
    for dataset in datasets:
        doc_id = dataset["id"]
        text = dataset["article_original"]
        gold = dataset["abstractive"]
        data.append((doc_id, text, gold))

    return data