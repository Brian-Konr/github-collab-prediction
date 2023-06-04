import networkx as nx
import numpy as np
import pandas as pd
import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
from torchmetrics import F1Score, ConfusionMatrix
from sklearn.metrics import classification_report, confusion_matrix, f1_score


GRAPH_FEATURES = [
    "pagerank",
    "hits",
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "resource_allocation_index",
    "jaccard_coefficient",
    "adamic_adar",
    "preferential_attachment",
    "common_neighbors",
    "common_neighbor_centrality",
]


# fix random seed
random.seed(42)
np.random.seed(42)


class GitHubCollabDataset(Dataset):
    def __init__(
        self,
        path: str = "../dataset/postprocess-data/collab_month_edges.pickle",
        emb_path: str = "../dataset/v2/user_embedding_v2.pkl",
        node_size: int = 5390,
        seq_len: int = 8,
        use_graph_feature: bool = False,
        only_graph_feature: bool = False,
    ):
        # 2023-04 as test set, and abandon 2023-05
        with open(path, "rb") as f:
            self.data = pickle.load(f)

        years = list(self.data.keys())

        years.remove("2023-04")
        years.remove("2023-05")

        graphs = dict()
        graph_feats = dict()

        # years = ["2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10", "2020-11"]
        print("Building graphs...")
        for year in tqdm(years):
            edges = self.data[year]
            G = nx.Graph()
            G.add_nodes_from(range(node_size))
            G.add_edges_from(edges)
            G.add_edges_from(edges)
            graphs[year] = G

            if not use_graph_feature:
                continue
            feat = dict()
            feat["pagerank"] = nx.pagerank(G)
            feat["hits_a"] = nx.hits(G)[0]
            feat["hits_h"] = nx.hits(G)[1]
            feat["degree_centrality"] = nx.degree_centrality(G)
            # feat["betweenness_centrality"] = nx.betweenness_centrality(G)
            feat["closeness_centrality"] = nx.closeness_centrality(G)
            # feat["katz_centrality"] = nx.katz_centrality_numpy(G)
            graph_feats[year] = feat

        with open(emb_path, "rb") as f:
            self.emb = pickle.load(f)

        subsequences = [years[i: i + seq_len + 1] for i in range(len(years) - seq_len)]

        self.x = list()
        self.y = list()

        print("Building dataset...")
        for subsequence in tqdm(subsequences):
            data2add = []

            for a in range(node_size):
                for b in range(node_size):
                    if a == b:
                        continue
                    if graphs[subsequence[-1]].has_edge(a, b) and random.random() < 0.5:
                        data2add.append((a, b, 1))
                    elif random.random() < 0.0001:
                        data2add.append((a, b, 0))
            for a, b, label in tqdm(data2add, leave=False):
                x_sub = torch.Tensor([])
                for year in subsequence[:-1]:
                    x_a, x_b, edge_feat = (
                        list(),
                        list(),
                        list(),
                    )
                    edge_feat += [int((a, b) in graphs[year].edges)]
                    if use_graph_feature:
                        for feat in graph_feats[year]:
                            x_a += [graph_feats[year][feat][a]]
                            x_b += [graph_feats[year][feat][b]]

                        edge_temp = list()
                        # the return is tuple, i only want to the third value
                        edge_temp += nx.resource_allocation_index(
                            graphs[year], [(a, b)]
                        )
                        edge_temp += nx.jaccard_coefficient(graphs[year], [(a, b)])
                        edge_temp += nx.adamic_adar_index(graphs[year], [(a, b)])
                        edge_temp += nx.preferential_attachment(graphs[year], [(a, b)])
                        # edge_feat += nx.cn_soundarajan_hopcroft(graphs[year], [(a, b)])
                        edge_temp += nx.common_neighbor_centrality(graphs[year], [(a, b)])
                        edge_temp = [i[2] for i in edge_temp]
                        edge_feat += edge_temp

                    if only_graph_feature:
                        x_t = torch.Tensor(x_a + x_b + edge_feat, dtype=torch.float32)
                        x_sub = torch.cat([x_sub, x_t])
                    else:
                        x_a = torch.FloatTensor(x_a)
                        x_b = torch.FloatTensor(x_b)
                        x_a = torch.cat([x_a, self.emb[year][a]])
                        x_b = torch.cat([x_b, self.emb[year][b]])
                        # print(edge_feat)
                        edge_feat = torch.FloatTensor(edge_feat)
                        x_t = torch.cat([x_a, x_b, edge_feat])
                        if len(x_sub) == 0:
                            x_sub = torch.unsqueeze(x_t, dim=0)
                        else:
                            x_sub = torch.cat([x_sub, torch.unsqueeze(x_t, dim=0)], dim=0)

                self.x.append(x_sub)
                self.y.append(label)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class GithubCollabTestDataset(Dataset):
    def __init__(
            self,
            path: str = "../dataset/postprocess-data/collab_month_edges.pickle",
            emb_path: str = "../dataset/v2/user_embedding_v2.pkl",
            node_size: int = 5390,
            seq_len: int = 8,
            use_graph_feature: bool = False,
            only_graph_feature: bool = False,
    ):
        with open(path, "rb") as f:
            self.data = pickle.load(f)

        years = ["2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04"]

        graphs = dict()
        graph_feats = dict()

        print("Building graphs...")
        for year in tqdm(years):
            edges = self.data[year]
            G = nx.Graph()
            G.add_nodes_from(range(node_size))
            G.add_edges_from(edges)
            G.add_edges_from(edges)
            graphs[year] = G

            if not use_graph_feature:
                continue
            feat = dict()
            feat["pagerank"] = nx.pagerank(G)
            feat["hits_a"] = nx.hits(G)[0]
            feat["hits_h"] = nx.hits(G)[1]
            feat["degree_centrality"] = nx.degree_centrality(G)
            # feat["betweenness_centrality"] = nx.betweenness_centrality(G)
            feat["closeness_centrality"] = nx.closeness_centrality(G)
            # feat["katz_centrality"] = nx.katz_centrality_numpy(G)
            graph_feats[year] = feat

        with open(emb_path, "rb") as f:
            self.emb = pickle.load(f)

        subsequences = [years]

        self.x = list()
        self.y = list()

        print("Building dataset...")
        for subsequence in tqdm(subsequences):
            data2add = []

            for a in range(node_size):
                for b in range(node_size):
                    if a == b:
                        continue
                    if graphs[subsequence[-1]].has_edge(a, b):
                        data2add.append((a, b, 1))
                    elif random.random() < 0.0005:
                        data2add.append((a, b, 0))
            for a, b, label in tqdm(data2add, leave=False):
                x_sub = torch.Tensor([])
                for year in subsequence[:-1]:
                    x_a, x_b, edge_feat = (
                        list(),
                        list(),
                        list(),
                    )
                    edge_feat += [int((a, b) in graphs[year].edges)]
                    if use_graph_feature:
                        for feat in graph_feats[year]:
                            x_a += [graph_feats[year][feat][a]]
                            x_b += [graph_feats[year][feat][b]]

                        edge_temp = list()
                        # the return is tuple, i only want to the third value
                        edge_temp += nx.resource_allocation_index(
                            graphs[year], [(a, b)]
                        )
                        edge_temp += nx.jaccard_coefficient(graphs[year], [(a, b)])
                        edge_temp += nx.adamic_adar_index(graphs[year], [(a, b)])
                        edge_temp += nx.preferential_attachment(graphs[year], [(a, b)])
                        # edge_feat += nx.cn_soundarajan_hopcroft(graphs[year], [(a, b)])
                        edge_temp += nx.common_neighbor_centrality(graphs[year], [(a, b)])
                        edge_temp = [i[2] for i in edge_temp]
                        edge_feat += edge_temp

                    if only_graph_feature:
                        x_t = torch.Tensor(x_a + x_b + edge_feat, dtype=torch.float32)
                        x_sub = torch.cat([x_sub, x_t])
                    else:
                        x_a = torch.FloatTensor(x_a)
                        x_b = torch.FloatTensor(x_b)
                        x_a = torch.cat([x_a, self.emb[year][a]])
                        x_b = torch.cat([x_b, self.emb[year][b]])
                        # print(edge_feat)
                        edge_feat = torch.FloatTensor(edge_feat)
                        x_t = torch.cat([x_a, x_b, edge_feat])
                        if len(x_sub) == 0:
                            x_sub = torch.unsqueeze(x_t, dim=0)
                        else:
                            x_sub = torch.cat([x_sub, torch.unsqueeze(x_t, dim=0)], dim=0)

                self.x.append(x_sub)
                self.y.append(label)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    dataset = GitHubCollabDataset(use_graph_feature=True)
    print(len(dataset))
    # print(dataset[0][0].shape)
    # print(Counter(dataset.y))
    # dataset = GithubCollabTestDataset(use_graph_feature=False)
    #
    # xs = dataset.x
    # y = dataset.y
    # pred = list()
    #
    # for x in xs:
    #     prev_label = x[-1, -1]
    #
    #     pred.append(prev_label)
    #
    # print(y, pred)
    # print(confusion_matrix(y, pred))
    # print(f1_score(y, pred))




