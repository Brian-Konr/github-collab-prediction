import transformers
import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name",
    type=str,
    choices=[
        "glove.6B.300d",
        "bert-base-uncased",
        "roberta-base",
        "sbert",
        "huggingface/CodeBERTa-small-v1",
        "all",
    ],
    default="roberta-base",
    help="model choice for extract embedding",
)

parser.add_argument(
    "--cluster_method",
    type=str,
    choices=[None, "kmeans", "dbscan", "all"],
    default=None,
    help="cluster method",
)

parser.add_argument(
    "--result_dir", type=str, default="postprocess_data", help="path to save result"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="batch size for extract embedding, may need to reduce size if cuda of of memory",
)

parser.add_argument(
    "--num_clusters",
    type=int,
    default=10,
    help="cluster number for kmeans if using clustering method, if -1, use elbow method to find best cluster number",
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)

parser.add_argument(
    "--strategy",
    type=str,
    choices=["mean", "cls", "max"],
    help="strategy for extract embedding from bert-like model",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed",
)

args = parser.parse_args()

# Set up random seed manually
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

df = pd.read_csv("../dataset/data/repos.csv")

trainset = list()

for i in range(len(df)):
    row = df.iloc[1709]
    data = (
        f"Repo name: {row['name'].split('/')[-1]}. Description: {row['description']}."
        f"Programming Language: {eval(row['languages'])[0] if eval(row['languages']) else 'Document'}. "
        f"Tag: {','.join(eval(row['topics'])) if eval(row['topics']) else 'Empty'}"
    )
    trainset.append(data)


if args.model_name == "all":
    raise NotImplementedError
elif os.path.exists(os.poth.join(args.result_dir, f"{args.model_name}_embeddings_{args.strategy}.npy")):
    print(f"Embedding for {args.model_name} already exists, skip")
    exit()
elif args.model_name == "glove.6b.300d":
    raise NotImplementedError
elif args.model_name in [
    "bert-base-uncased",
    "roberta-base",
    "huggingface/CodeBERTa-small-v1",
]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    model.eval()
    model.to(args.device)

    for i in range(0, len(trainset), args.batch_size):
        batch = (
            trainset[i : i + args.batch_size]
            if i + args.batch_size <= len(trainset)
            else trainset[i:]
        )

        inputs = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)

        if args.strategy == "mean":
            attention_mask = inputs.attention_mask.unsqueeze(-1)
            embeddings = (
                torch.sum(outputs.last_hidden_state * attention_mask, dim=1)
                / torch.sum(attention_mask, dim=1).numpy()
            )
        elif args.strategy == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        else:
            raise NotImplementedError("Invalid Strategy")

        if i == 0:
            all_embeddings = embeddings
        else:
            all_embeddings = np.concatenate((all_embeddings, embeddings), axis=0)

    with open(
        os.path.join(args.result_dir, f"{args.model_name}_embeddings_{args.strategy}.npy"), "wb"
    ) as f:
        np.save(f, all_embeddings)

elif args.model_name == "sbert":
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(trainset)

    with open(
        os.path.join(args.result_dir, f"{args.model_name}_embeddings_{args.strategy}.npy"), "wb"
    ) as f:
        np.save(f, embeddings)
