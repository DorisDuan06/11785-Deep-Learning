import numpy as np
from collections import Counter

'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''
def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)
    seq_len = y_probs.shape[1]
    SymbolSets = [""] + SymbolSets
    y_probs = np.mean(y_probs, axis=2)

    forward_path, forward_prob = [], 1
    for t in range(seq_len):
        if t > 0:
            y_probs[:, t] *= forward_prob
        i = np.argmax(y_probs[:, t])
        forward_prob = y_probs[i, t]
        if len(forward_path) == 0 or SymbolSets[i] != forward_path[-1]:
            forward_path.append(SymbolSets[i])
        prev_symbol = SymbolSets[i]
    forward_path = "".join(forward_path)
    return (forward_path, forward_prob)


##############################################################################


def prune(pathsSymbol, pathsBlank, BeamWidth):
    scores = []
    for _, v in pathsSymbol.items():
        scores.append(v)
    for _, v in pathsBlank.items():
        scores.append(v)
    scores = sorted(scores, reverse=True)
    cutoff = scores[BeamWidth-1] if BeamWidth < len(scores) else scores[-1]

    prunedPathsSymbol, prunedPathsBlank = {}, {}
    for k, v in pathsSymbol.items():
        if v >= cutoff:
            prunedPathsSymbol[k] = v
    for k, v in pathsBlank.items():
        if v >= cutoff:
            prunedPathsBlank[k] = v
    return prunedPathsSymbol, prunedPathsBlank


'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''
def BeamSearch(SymbolSets, y_probs, BeamWidth):
    # Follow the pseudocode from lecture to complete beam search :-)
    seq_len = y_probs.shape[1]
    y_probs = np.mean(y_probs, axis=2)

    # Initialize paths
    newPathsSymbol, newPathsBlank = {}, {}
    for i, c in enumerate(SymbolSets):
        newPathsSymbol[c] = y_probs[i+1, 0]
    newPathsBlank[""] = y_probs[0, 0]

    for t in range(1, seq_len):
        # Prune down to BeamWidth
        pathsSymbol, pathsBlank = prune(newPathsSymbol, newPathsBlank, BeamWidth)

        # Extend paths by a blank
        newPathsBlank = {}
        for path, score in pathsBlank.items():
            newPathsBlank[path] = score * y_probs[0, t]
        for path, score in pathsSymbol.items():
            newPathsBlank[path] = newPathsBlank.get(path, 0) + score * y_probs[0, t]

        # Extend paths by a symbol
        newPathsSymbol = {}
        for path, score in pathsBlank.items():
            for i, c in enumerate(SymbolSets):
                newpath = path + c
                newPathsSymbol[newpath] = score * y_probs[i+1, t]
        for path, score in pathsSymbol.items():
            for i, c in enumerate(SymbolSets):
                newpath = path + c if c != path[-1] else path
                newPathsSymbol[newpath] = newPathsSymbol.get(newpath, 0) + score * y_probs[i+1, t]

    # Merge identical paths
    mergedPathScores = newPathsSymbol.copy()
    for path, score in newPathsBlank.items():
        mergedPathScores[path] = mergedPathScores.get(path, 0) + score
    bestPath = max(mergedPathScores, key=mergedPathScores.get)
    return (bestPath, mergedPathScores)
