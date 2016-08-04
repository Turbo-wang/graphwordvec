import numpy
import types
import networkx as nx


max_ = 0
V = {1:[1,0,1], 2:[0,1,0], 3:[1,1,1]}
print V.items()
g = nx.Graph()
for key, value in V.items():
    g.add_node(key)
    for sub_key, sub_value in enumerate(value):
        if sub_key+1 != key and sub_value == 1:
            g.add_edge(key, sub_key+1)

def neighbor(U, i):
    Ner = {}
    li = U[i]
    for key, value in enumerate(li):
        if key+1 != i and value == 1:
            Ner.update({key+1:U[key+1]})
    return Ner

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


def extract_clique(U, size):
    print U
    global max_
    if len(U) == 0:
        if size > max_:
            max_ = size
            print U.nodes()
        return
    while len(U) != 0:
        if size + len(U) <= max_:
            return
        i = min(U.nodes())
        U.remove_node(i)
        UN = U.copy()
        UN.add_nodes_from(g.neighbors(i))
        extract_clique(UN, size + 1)
    return



def old():
    

    b = g.copy()
    print b.nodes()
    print b.edges()
    list(nx.clique.find_cliques_recursive(b))

if __name__ == '__main__':
    old()