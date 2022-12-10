# @Time    : 2022/10/25  17:17
# @Auther  : Teng Zhang
# @File    : model_define.py
# @Project : ASTL
# @Software: PyCharm

import numpy as np
import torch
import Result_evalute
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
plt.rcParams['font.sans-serif'] = ['Times New Roman']

def acc_pre(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae

class SourceCNN(nn.Module):
    def __init__(self):
        super(SourceCNN, self).__init__()

        self.FC1 = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.FC3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.FC4 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.predict = nn.Linear(16, 6)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.FC1(x)
        x = self.relu(x)

        x = self.FC2(x)
        middle_1 = self.relu(x)

        x = self.FC3(middle_1)
        middle_2 = self.relu(x)

        x = self.FC4(middle_2)
        middle_3 = self.relu(x)

        result = self.predict(middle_3)

        return result

class TargetCNN(nn.Module):
    def __init__(self):
        super(TargetCNN, self).__init__()

        self.FC1 = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.FC3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.FC4 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.predict = nn.Linear(16, 6)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.FC1(x)
        middle_0 = self.relu(x)

        x = self.FC2(x)
        middle_1 = self.relu(x)

        x = self.FC3(middle_1)
        middle_2 = self.relu(x)

        x = self.FC4(middle_2)
        middle_3 = self.relu(x)

        result = self.predict(middle_3)

        middle_result = list([middle_0, middle_1, middle_2, middle_3])
        return middle_result, result

    def forward_pre(self, x):

        x = self.FC2(x)
        middle_1 = self.relu(x)

        x = self.FC3(middle_1)
        middle_2 = self.relu(x)

        x = self.FC4(middle_2)
        middle_3 = self.relu(x)

        result = self.predict(middle_3)

        middle_result = list([middle_1, middle_2, middle_3])
        return middle_result, result

if __name__ == '__main__':
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
    target_x = torch.Tensor(target_x).to(DEVICE)
    target_y = torch.Tensor(error_M).to(DEVICE)

    Source = SourceCNN().to(DEVICE)
    learning_rate = 1e-2
    regularization = 1e-5
    num_epochs = 3000
    Batch_size = 300

    optimizer = torch.optim.Adam(Source.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = torch.nn.MSELoss()
    torch_dataset = Data.TensorDataset(source_x, source_y)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=Batch_size, shuffle=True)

    loss_list = []
    accuracy_list = []
    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            prediction = Source.forward(batch_x)
            Loss = criterion(prediction, batch_y)
            loss_list.append(Loss.data / len(batch_x))  
            acc = acc_pre(batch_y.cpu().data.numpy(), prediction.cpu().data.numpy())
            accuracy_list.append(acc)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print(epoch, Loss.data)
    source_ypre = Source.forward(source_x)
    torch.cuda.empty_cache()
    print('Results on simulation domain(All dimension):')
    Result_evalute.predict(source_y.data.numpy(), source_ypre.data.numpy())

    torch.save(Source.state_dict(), "model/Simulation.pth")

    plt.subplot(121)
    plt.plot(accuracy_list, 'r', linewidth=2, label='accuracy')
    plt.xlim([-1, num_epochs])
    plt.ylim(0, np.max(accuracy_list))
    plt.xlabel('Epoch')
    plt.ylabel('MAE/mm')
    plt.grid('on')

    plt.subplot(122)
    plt.plot(loss_list, 'b', linewidth=2, label='accuracy')
    plt.xlim([-1, num_epochs])
    plt.ylim([0, np.max(loss_list)])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.grid('on')
    plt.show()

    plt.figure(1)
    error_x = source_y.data.numpy()[:,0]
    error_x_pre = source_ypre.data.numpy()[:,0]
    print('Results on simulation domain(1st dimension):')
    Result_evalute.predict_i(source_y.data.numpy(), source_ypre.data.numpy(), 0)
    plt.scatter(error_x, error_x_pre, s=36, label='error_x')

    error_x = source_y.data.numpy()[:,1]
    error_x_pre = source_ypre.data.numpy()[:,1]
    print('Results on simulation domain(2st dimension):')
    Result_evalute.predict_i(source_y.data.numpy(), source_ypre.data.numpy(),1)
    plt.scatter(error_x, error_x_pre, s=36, label='error_y')

    error_x = source_y.data.numpy()[:,2]
    error_x_pre = source_ypre.data.numpy()[:,2]
    print('Results on simulation domain(3st dimension):')
    Result_evalute.predict_i(source_y.data.numpy(), source_ypre.data.numpy(), 2)
    plt.scatter(error_x, error_x_pre, s=36, label='error_z')

    error_x = source_y.data.numpy()[:,3]
    error_x_pre = source_ypre.data.numpy()[:,3]
    print('Results on simulation domain(4st dimension):')
    Result_evalute.predict_i(source_y.data.numpy(), source_ypre.data.numpy(), 3)
    plt.scatter(error_x, error_x_pre, s=36, label='error_rx')

    error_x = source_y.data.numpy()[:,4]
    error_x_pre = source_ypre.data.numpy()[:,4]
    print('Results on simulation domain(5st dimension):')
    Result_evalute.predict_i(source_y.data.numpy(), source_ypre.data.numpy(), 4)
    plt.scatter(error_x, error_x_pre, s=36, label='error_ry')


    error_x = source_y.data.numpy()[:,5]
    error_x_pre = source_ypre.data.numpy()[:,5]
    print('Results on simulation domain(6st dimension):')
    Result_evalute.predict_i(source_y.data.numpy(), source_ypre.data.numpy(), 5)
    plt.scatter(error_x, error_x_pre, s=36, label='error_rz')
    
    min_value = min([min(source_y.data.numpy()[:,0]),min(source_y.data.numpy()[:,1]),min(source_y.data.numpy()[:,2])])
    max_value = max([max(source_y.data.numpy()[:,0]),max(source_y.data.numpy()[:,1]),max(source_y.data.numpy()[:,2])])
    
    plt.plot((-2.5,2.5),(-2.5,2.5),label="Reference Line")
    plt.legend(fontsize=12)
    plt.grid('on')

    plt.xlabel('True value(mm/°)', fontsize=13)
    plt.ylabel('Predict value(mm/°)', fontsize=13)

    plt.show()
