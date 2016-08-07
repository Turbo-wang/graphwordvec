import numpy as np
import os
from scipy.spatial.distance import cosine
import sys


def build_dict(fname, words=None, normalize=False):
    id2row = []
    word_dict = {}
    word_list = []
    with open(fname) as f:
        for i, line in enumerate(f):
            word = line.split()[0]
            if i != 0 and (words is None or word in words):
                id2row.append(word)
                word_list.append(line.split()[1:])
    #normalize
    word_list = np.asarray(word_list, dtype='float32')
    if normalize:
        row_norms = np.sqrt(np.multiply(word_list, word_list).sum(1))
        row_norms = np.reshape(row_norms, [-1, 1])
        row_norms = row_norms.astype(np.double)
        row_norms[row_norms != 0] = np.array(1.0/row_norms[row_norms != 0]).flatten()
        word_list = np.multiply(word_list, row_norms)
    # for word, vector in zip(id2row, word_list):
    #     word_dict[word] = vector
    # np.savetxt("enwordnormal", word_list)
    return  word_list, id2row


def similarity(word_list, id2row):
    test_file = 'word-test.v1.txt'
    test_list = []
    for word, vector in zip(id2row, word_list):
        word_dict[word] = vector
    with open(os.path.join('../corpus', test_file)) as f:
        for line in f:
            abcd = line.strip().split()
            test_list.append(abcd)

    total = len(test_list)
    right = 0
    for test in test_list:
        vec1 = test[0]
        vec2 = test[1]
        vec3 = test[2]
        vec4 = test[3]
        vec_real = vec1 - vec2 + vec3
        dist = sys.maxsize
        vec_cand = []
        for word in word_list:
            vector = word_dict[word]
            curr_dist = cosine(vec_cand, vector)
            if curr_dist < dist:
                dist = curr_dist
                vec_cand = vector
        if np.array_equal(vec_cand, vec_real):
            right += 1
    print(right)