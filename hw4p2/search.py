import numpy as np


def GreedySearch(y_probs, index2letter):
    # (N, T, 35)
    seq_len = y_probs.shape[1]

    forward_paths = []
    for y_prob in y_probs:  # (T, 35)
        forward_path = ""
        for t in range(seq_len):
            ind = np.argmax(y_prob[t])
            if index2letter[ind] == '<eos>':
                break
            forward_path += index2letter[ind]
        forward_paths.append(forward_path)
    return forward_paths
