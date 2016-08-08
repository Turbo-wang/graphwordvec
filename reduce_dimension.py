import nltk
import numpy as np
import networkx as nx
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from itertools import combinations
# import matplotlib.pyplot as plt
import types
import string
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from sklearn import manifold
import os
import time
import pickle


manifold_method = ["isomap", "LocallyLinearEmbedding", "MDS", "SpectralEmbedding",
         "TSNE", "modified_LLE", "HessianLLE", "LTSA"]

def get_txt_line(file_name):
    translator = str.maketrans({key: None for key in string.punctuation})
    with open(file_name) as f:
        for line in f:
            line = line.strip().lower()
            line = line.translate(translator)
            sen = nltk.tokenize.wordpunct_tokenize(line)
            yield sen


def sents_list(sents):
    sentences_list = []
    for sen in sents:
        if sen != []:
            sentences_list.append(sen)
    # print(len(sentences_list))
    return sentences_list


def count_word_fre(sents_list):
    sents = []
    for sen in sents_list:
        sents += sen
    word_fre = Counter(sents)
    return word_fre


def build_graph(file_name):
    sentences_list = sents_list(get_txt_line(file_name))
    G = nx.Graph()
    word_fre = count_word_fre(sentences_list)
    for sent in sentences_list:
        H = nx.Graph()
        edges = combinations(sent, 2)
        H.add_edges_from(edges)
        for edge in H.edges(): 
            word1 = edge[0]
            word2 = edge[1]
            PMI_det = 1 / (word_fre[word1] * word_fre[word2])
            if G.has_edge(word1, word2):
                G[word1][word2]['weight'] += PMI_det
            else:
                G.add_edge(word1, word2, weight=PMI_det)
    return G


def PMI_filter(G, pmi_threshhold):
    for edge in G.edges():
        if G[edge[0]][edge[1]]['weight'] < pmi_threshhold:
            G.remove_edge(edge[0], edge[1])
    

def display_edges(G):
    for edge in G.edges():
        print('edges', edge, "pmi", G[edge[0]][edge[1]]['weight'])
    

def find_all_cliques(G):
    cliques = list(nx.clique.find_cliques(G))
    return cliques


def save_graph(G, path):
    nx.write_yaml(G, os.path.join('./corpus', path))


def find_nodes_cliques(G, nodes=None):
    H = G.copy()
    cliques_list = []
    nodes_list = []
    if not isinstance(nodes, list) and nodes != None:
        nodes_list.append(nodes)
    elif nodes != None:
        nodes_list = nodes
    neighbor_list = []
    for node in nodes_list:
        neighbor_list += G.neighbors(node)
    neighbor_list += nodes_list
    all_nodes = H.nodes()
    remove_nodes = []
    for node in all_nodes:
        if node not in neighbor_list:
            remove_nodes.append(node)
    H.remove_nodes_from(remove_nodes)
    return list(nx.clique.find_cliques(H))


def build_matrix(cliques):
    words_set = set()
    for key in cliques:
        words_set.update(key)
    words_list = sorted(list(words_set))
    cliques_num = len(cliques)
    words_num = len(words_set)
    cli_matrix = []
    for key in cliques:
        line = []
        for word in words_list:
            if word in key:
                line.append(1)
            else:
                line.append(0)
        cli_matrix.append(line)
    return words_list, cli_matrix


def _pca_dimension(cli_matrix, dimension):
    cli_matrix = cli_matrix.T
    s = np.sum(cli_matrix, axis = 1) / len(cli_matrix[0])
    s = np.reshape(s, [len(cli_matrix),1])
    cli_matrix = cli_matrix - s
    pca = PCA(n_components=dimension)
    X = pca.fit_transform(cli_matrix)
    return X


def pca_reduct_dimension(cli_list, dimension):
    words_list, cli_matrix = build_matrix(cli_list)
    cli_matrix = np.asarray(cli_matrix, dtype='int32')
    di_matrix = _pca_dimension(cli_matrix, dimension)
    return words_list, di_matrix


def ca_dimension():
    pass


def nn_dimension(cli_list, dimension, win, sg, hs):
    sentences = []
    for cli in cli_list:
        sentences.append(cli)
    model =  Word2Vec(sentences, size=dimension, min_count = 1, window=win, sg = sg , hs = hs, workers = 4 , iter = 5)
    model.save_word2vec_format("word2vec_%sd_win%s_sg%s_hs%s.out" % (str(dimension), str(win), str(sg), str(hs)), binary=False) 
    return "word2vec_%sd_win%s_sg%s_hs%s.out" % (str(dimension), str(win), str(sg), str(hs))


def manifold_reduce_dimension(name, dimension, cli_list):
    words_list, cli_matrix = build_matrix(cli_list)
    cli_matrix = np.asarray(cli_matrix, dtype='float64')
    di_matrix = _manifold_dimension(name, dimension, cli_matrix.T)
    return words_list, di_matrix


def _manifold_dimension(name, dimension, cli_matrix):
    n_neighbors = 3
    di_matrix = []
    if name == "isomap":
        X_iso = manifold.Isomap(n_neighbors, n_components=dimension).fit_transform(X)
        di_matrix = X_iso
    elif name == "LocallyLinearEmbedding":
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dimension,
                                      method='standard')
        X_lle = clf.fit_transform(cli_matrix) 
        di_matrix = X_lle       
    elif name == "MDS":
        clf = manifold.MDS(n_components=dimension, n_init=1, max_iter=100)
        X_mds = clf.fit_transform(cli_matrix)
        di_matrix = X_mds
    elif name == "SpectralEmbedding":
        embedder = manifold.SpectralEmbedding(n_components=dimension, random_state=0,
                                      eigen_solver="arpack")
        X_se = embedder.fit_transform(cli_matrix)
        di_matrix = X_se
    elif name == "TSNE":
        tsne = manifold.TSNE(n_components=dimension, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(cli_matrix)
        di_matrix = X_tsne
    elif name == "modified_LLE":
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dimension,
                                      method='modified')
        X_mlle = clf.fit_transform(cli_matrix)
        di_matrix = X_mlle
    elif name == "HessianLLE":
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dimension,
                                      method='hessian')
        X_hlle = clf.fit_transform(cli_matrix)
        di_matrix = X_hlle
    elif name == "LTSA":
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dimension,
                                      method='ltsa')
        X_ltsa = clf.fit_transform(cli_matrix)
        di_matrix = X_ltsa
    return di_matrix


if __name__ == '__main__':
    s = time.time()
    print(s)
    G = build_graph('corpus/wiki.en.text')
    # G = build_graph('test.txt')
    print(time.time() - s)
    save_graph(G, 'graph_no_pmi_en_wiki')
    PMI_filter(G, 0.0003)
    cli_list = find_all_cliques(G)
    with open('corpus/cli_PMI_0.0003', 'wb') as f:
        pickle.dump(cli_list, f)

    # pca_reduct_dimension(cli_list, 2)
    # nn_dimension(cli_list, 2, win=5, sg = 0, hs = 0)
    # pca_dimension(cli_matrix, 2)
    # x = manifold_reduce_dimension("LTSA", 2, cli_list)
    # print(x)