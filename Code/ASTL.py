# @Time    : 2022/10/31  17:15
# @Auther  : Teng Zhang
# @File    : ASTL.py
# @Project : ASTL
# @Software: PyCharm



import MGS
import numpy as np
import pandas as pd
import torch

import Result_evalute
import CDA
import MDA
import model_define
import NewDataGen
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.manifold import TSNE
import warnings
import datetime

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_prepare(number_choose):
    """:param
    number_choose
    sample_startage
    """

    theta_S = pd.read_csv("theta.csv").values[:, 0:6]
    error_S = pd.read_csv("Error_S.csv").values[:, 0:6]

    theta_M = pd.read_csv("theta.csv").values[:, 0:6]
    error_M = pd.read_csv("Error_M.csv").values[:, 0:6]

    standardScaler = StandardScaler().fit(theta_S)
    source_x = standardScaler.transform(theta_S)
    source_x = torch.Tensor(source_x).to(DEVICE)
    source_y = torch.Tensor(error_S).to(DEVICE)

    standardScaler = StandardScaler().fit(theta_M)
    target_x = standardScaler.transform(theta_M)

    [_, _, index] = MGS.MGS_offer(target_x, error_M, number_choose)

    target_x = torch.Tensor(target_x).to(DEVICE)
    target_y = torch.Tensor(error_M).to(DEVICE)

    index1 = index
    index2 = np.delete(np.arange(len(target_x)), index1)
    t_xseed = target_x[index1, :]
    t_yseed = target_y[index1, :]
    t_xtest = target_x[index2, :]
    t_ytest = target_y[index2, :]
    return source_x, source_y, t_xseed, t_yseed, t_xtest, t_ytest, index1


def load_TargetCNN(model, name):

    pretrained_dict = torch.load("model/Simulation.pth")

    new_dict = model.state_dict()
    pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in new_dict}
    new_dict.update(pretrained_dict1)
    model.load_state_dict(new_dict)

    if name == 'ASTL':
        namelist = ['predict.bias', 'predict.weight',
                    'FC4.0.bias', 'FC4.0.weight',
                    'FC3.0.bias', 'FC3.0.weight',
                    'FC2.0.bias', 'FC2.0.weight']

        for name, value in model.named_parameters():
            if name in namelist:
                value.requires_grad = True
            else:
                value.requires_grad = False
    else:
        namelist = []
        for name, value in model.named_parameters():
            if name not in namelist:
                value.requires_grad = True
            else:
                value.requires_grad = False

    print('Moldel loading finished')

def test_TargetCNN(model, t_xtest):
    with torch.no_grad():
        inter_x, ypre = model.forward(t_xtest)
    for i in range(len(inter_x)):
        inter_x[i] = inter_x[i].cpu().data.numpy()
    ypre = ypre.cpu().data.numpy()
    return inter_x, ypre


def F_Fc_ASTL(model, t_xseed, t_yseed, t_xtest, source_x, source_y, epoch, learning_rate, regularization):
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=regularization)
    criterion = torch.nn.MSELoss()
    for i in range(epoch):
        t_seed_list, prediction1 = model.forward(t_xseed)
        with torch.no_grad():
            t_test_list, prediction4 = model.forward(t_xtest)

        t_seed = torch.cat((t_seed_list[0], t_yseed), 1)
        with torch.no_grad():
            source_list, prediction2 = model.forward(source_x)
        s = torch.cat((source_list[0], source_y), 1)
        new_middle_test, label_pred, Residual = NewDataGen.CLUSTER(s, t_seed)
        new_source_liketarget = torch.from_numpy(new_middle_test).to(DEVICE)
        feartue = new_source_liketarget[:, 0:(np.shape(new_source_liketarget)[1] - 6)]
        label = new_source_liketarget[:, new_source_liketarget.shape[1] - 6:new_source_liketarget.shape[1]].reshape(
            np.shape(new_source_liketarget)[0], 6)
        new_source_list, prediction3 = model.forward_pre(feartue)
        CEOD_loss = CDA.forward(t_test_list, prediction4, t_seed_list, t_yseed)
        MMD = MDA.MMD_loss()
        MMDloss = MMD.forward(feartue, t_seed_list[0])
        Loss = 2 * (criterion(prediction3, label) + criterion(prediction1, t_yseed)) + 10 * (0.2 * CEOD_loss + 0.8 * MMDloss)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(i, Loss.data)

def HighDemention_selected(model, source_x, target_x):
    with torch.no_grad():
        source_list, prediction2 = model.forward(source_x)
        target_list, prediction5 = model.forward(target_x)
    s_x = source_list[0]
    t_x = target_list[0]
    return s_x, t_x

if __name__ == '__main__':

    num = 30  # 表示选num个样本
    nRepeat = 10  # 表示每一个策略，循环nRepeat次
    model_num = 1

    Result = np.zeros((nRepeat, 2 * model_num))
    index_hold = np.zeros((nRepeat, num))
    for r in range(nRepeat):
        source_x, source_y, t_xseed, t_yseed, t_xtest, t_ytest, index = data_prepare(num)

        if 1:
            name = 'ASTL'
            model_ASTL = model_define.TargetCNN().to(DEVICE)
            load_TargetCNN(model_ASTL, name)
            learning_rate = 3e-4
            regularization = 1e-4
            epoch = 100
            F_Fc_ASTL(model_ASTL, t_xseed, t_yseed, t_xtest, source_x, source_y, epoch, learning_rate,
                           regularization)
            ASTL_Xtest, ASTL_ytest_pre = test_TargetCNN(model_ASTL, t_xtest)
            print('Results of Stratage', i + 1, 'model (', name, ') :')
            result_ASTL = Result_evalute.predict(t_ytest.cpu().data.numpy(), DTRSR_ytest_pre)



        Result[r, :] =result_DTRSR
        index_hold[r, :] = index
    now = datetime.datetime.now()
    name = model_num * ['MAE', 'RMSE']
    principle = pd.DataFrame(columns=name, data=Result)
    principle.to_csv('Result/' + choose_stratage[i] + str(num) + '_' + now.strftime("%H-%M-%S") + '.csv')

    principle2 = pd.DataFrame(data=index_hold)
    principle2.to_csv('Index/' + choose_stratage[i] + str(num) + '_' + now.strftime("%H-%M-%S") + '.csv')
