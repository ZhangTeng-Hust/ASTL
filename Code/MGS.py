# @Time    : 2022/7/9  8:21
# @Auther  : Teng Zhang
# @File    : MGS.py
# @Project : ASTL
# @Software: PyCharm

from scipy.spatial.distance import pdist, squareform
import numpy as np
import random
import math
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

def MGS_offer(input, label, num_target):
    ''':param
    input: 输入特征
    label: 输出标签
    num_target: 所需要的数目
    '''

    num_target = int(num_target)
    minN_int = int(6)
    maxN_int = int(num_target)

    X0 = input
    Y0 = label.reshape((len(input), 6))
    numY0 = len(Y0)

    ids = np.array(random.sample(list(np.linspace(0, numY0 - 1, numY0)), math.ceil(0.8 * numY0))).astype(np.int32)

    X = X0[ids, :]
    Y = Y0[ids, :]
    numY = len(Y)

    id_middle = np.array(random.sample(list(np.linspace(0, numY - 1, numY)), num_target)).astype(np.int32)

    idsTrain = np.tile(id_middle, (1, 1))
    distX = squareform(pdist(X))
    for n in np.linspace(minN_int, maxN_int-1, (maxN_int - minN_int)):
        if n == minN_int:
            dist = np.mean(distX, axis=1)
            midd = np.array(np.where(dist == np.min(dist))).astype(np.int32)
            idsTrain[0, 0] = midd[0, 0]
            idsTest = np.linspace(0, numY - 1, numY).astype(np.int32)  
            idsTest = np.delete(idsTest, idsTrain[0, 0])  
            for i in range(int(n) - 1):
                i = i + 2
                id_middle = np.linspace(0, i - 2, i - 1).astype(np.int32)

                middle_id = idsTrain[0, id_middle].reshape(len(id_middle), 1)
                dist = np.min(distX[idsTest, middle_id].transpose(), axis=1)
                idx = np.array(np.where(dist == np.max(dist))).astype(np.int32)
                idx = idx[0, 0]
                idsTrain[0, i - 1] = idsTest[idx] 
                idsTest = np.delete(idsTest, idx)

        id_selected = np.linspace(0, int(n) - 1, int(n)).astype(np.int32)
        sample_select = idsTrain[0, np.linspace(0, int(n) - 1, int(n)).astype(np.int32)]
        X_train = X[sample_select, :]
        Y_train = Y[sample_select, :]
        model = Ridge(alpha=0.6)
        wrapper = MultiOutputRegressor(model)
        wrapper.fit(X_train, Y_train)

        idsTest = np.linspace(0, numY - 1, numY).astype(np.int32)
        idsTest = np.delete(idsTest, idsTrain[0, id_selected])
        Y4 = Y.copy()

        X_test = X[idsTest, :]
        Y_test_pre = wrapper.predict(X_test)
        Y4[idsTest] = Y_test_pre

        distY = np.zeros((numY - int(n), int(n)))
        for i in range(int(n)):
            aaa = Y4[idsTest] - Y[idsTrain[0, i]] * np.ones((numY - int(n), 6))
            distY[:, i] = np.abs(aaa[:, 0] * aaa[:, 1] * aaa[:, 2] * aaa[:, 3] * aaa[:, 4] * aaa[:, 5])

        id = np.linspace(0, int(n) - 1, int(n)).astype(np.int32)
        id_reshape = idsTrain[0, id].reshape(len(id), 1)

        part1 = distX[idsTest, id_reshape].transpose()

        part2 = np.multiply(part1, distY)
        dist = np.min(part2, axis=1)
        idx = np.array(np.where(dist == np.max(dist))).astype(np.int32)
        idx = idx[0,0]
        idsTrain[0, int(n)] = idsTest[idx]

    index = idsTrain[0, :]
    X_selected = input[index, :]
    y_selected = label[index, :]
    return X_selected, y_selected, index