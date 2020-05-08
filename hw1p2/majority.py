import csv
import numpy as np
from scipy import stats

if __name__ == '__main__':
    ids = np.genfromtxt('70507.csv', delimiter=',')[1:,0]
    pred1 = np.genfromtxt('70507.csv', delimiter=',')[1:,1]
    pred1 = pred1.reshape(1, pred1.shape[0]).astype(int)
    # pred2 = np.genfromtxt('p_test.csv', delimiter=',')[1:,1]
    # pred2 = pred2.reshape(1, pred2.shape[0]).astype(int)
    pred3 = np.genfromtxt('81086.csv', delimiter=',')[1:,1]
    pred3 = pred3.reshape(1, pred3.shape[0]).astype(int)
    pred4 = np.genfromtxt('75869.csv', delimiter=',')[1:,1]
    pred4 = pred4.reshape(1, pred4.shape[0]).astype(int)
    pred5 = np.genfromtxt('yifan_72.csv', delimiter=',')[1:,1]
    pred5 = pred5.reshape(1, pred5.shape[0]).astype(int)
    pred6 = np.genfromtxt('yifan_76.csv', delimiter=',')[1:,1]
    pred6 = pred6.reshape(1, pred6.shape[0]).astype(int)
    pred = np.vstack((pred1, pred3, pred4, pred5, pred6))
    combined, _ = stats.mode(pred)
    combined = combined.flatten()

    with open('vote.csv', "w", newline='') as f:
        file = csv.writer(f, delimiter=',')
        file.writerow(["Id", "Category"])
        for i, c in enumerate(combined):
            file.writerow([ids[i], c])
