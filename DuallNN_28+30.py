import torch
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("Python") 

import matplotlib.pyplot as plt
import keyboard

def label_transform_reverse_tensor(x): #翻转预处理
    x = (torch.exp(x)-1 ) /10000
    return(x)

def standard_transform(x):
    # 计算每个特征的均值
    mean = x.mean(dim=0)
    # 计算每个特征的标准差
    std = x.std(dim=0, unbiased=False)  # unbiased=False 相当于 numpy 的 ddof=0
    # 避免使用0的标准差
    std[std == 0] = 1
    # 进行标准化转换
    x = (x - mean) / std
    return x,mean,std

def inverse_transform(x, mean, std):
    # 将标准化后的张量乘以标准差，然后加上均值
    x = x * std + mean
    return x

# 准备数据
X1 = np.load("Python\optim\DataFromBPTK\plasma28+urine15x2\plasma28+urine15x2_zzc.npy")  #输入数据
X2 = np.load("Python\optim\DataFromBPTK\plasma28+urine15x2\plasma28+urine15x2_SG.npy")  #输入数据

y1 = np.load("Python\optim\DataFromBPTK\labels_zzc.npy")
y2 = np.load("Python\optim\DataFromBPTK\labels_SG.npy")

X =  np.vstack((X1,X2))
y =  np.vstack((y1,y2))

# 数据预处理 标准化数据
X = torch.tensor(X).float()
y = torch.tensor(y).float()

X,X_mean,X_std = standard_transform(X)

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.15, random_state=20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

# 将数据转移到 GPU（如果可用

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)



class ResNetBlock(nn.Module):
    def __init__(self, hyperparas):
        super(ResNetBlock, self).__init__()
        
        self.hidden_dim = hyperparas['hidden_dim']
        self.block_layer_nums =hyperparas['block_layer_nums']
            
        # Define layers for the function f (MLP)
        self.layers = nn.ModuleList()
        
        for _ in range(self.block_layer_nums - 1):  # -2 because we already added one layer and last layer is already defined
            self.layers.append(nn.Linear(self.hidden_dim,self.hidden_dim ))
        
        # Layer normalization
        self.layernorms = nn.ModuleList()
        for _ in range(self.block_layer_nums - 1):  # -1 because layer normalization is not applied to the last layer
            self.layernorms.append(nn.LayerNorm(self.hidden_dim))
        
    def forward(self, x):
        # Forward pass through the function f (MLP)
        out = x
        for i in range(self.block_layer_nums - 1):  # -1 because last layer is already applied outside the loop
            out = self.layers[i](out)
            out = self.layernorms[i](out)
            out = torch.relu(out)
        
        # Element-wise addition of input x and output of function f(x)
        out = x + out
        
        return out
    
class ResNN_Forward(nn.Module):
    def __init__(self,hyperparas):
        super().__init__()
        self.input_dim = hyperparas['input_dim'] #28
        self.hidden_dim = hyperparas['hidden_dim'] #30
        self.hidden_nums = hyperparas['hidden_nums'] #3
        self.output_dim = hyperparas['output_dim'] #3
        self.block_layer_nums = hyperparas['block_layer_nums'] #3

        self.layer_list = []
        self.layer_list.append(nn.Sequential(nn.Linear(self.input_dim,self.hidden_dim),nn.ReLU() ) )

        for _ in range(self.hidden_nums-1):
            self.layer_list.append(ResNetBlock(hyperparas))

        self.layer_list.append(nn.Linear(self.hidden_dim,self.output_dim))

        self.linear_Res_final = nn.Sequential(*self.layer_list)

    def forward(self,inputs):
        
        return self.linear_Res_final(inputs)
    
class ResNN_Reverse(nn.Module):
    def __init__(self,hyperparas):
        super().__init__()
        self.input_dim = hyperparas['input_dim'] #28
        self.hidden_dim = hyperparas['hidden_dim'] #30
        self.hidden_nums = hyperparas['hidden_nums'] #3
        self.output_dim = hyperparas['output_dim'] #3
        self.block_layer_nums = hyperparas['block_layer_nums'] #3

        self.layer_list = []
        self.layer_list.append(nn.Sequential(nn.Linear(self.input_dim,self.hidden_dim),nn.ReLU() ) )

        for _ in range(self.hidden_nums-1):
            self.layer_list.append(ResNetBlock(hyperparas))

        self.layer_list.append(nn.Linear(self.hidden_dim,self.output_dim))

        self.linear_Res_final = nn.Sequential(*self.layer_list)

    def forward(self,inputs):
        
        return self.linear_Res_final(inputs)

#超参数合集
hyperparas_reverse = {'input_dim':58,'hidden_dim':30,'hidden_nums':3,'output_dim':3,'block_layer_nums':3}
hyperparas_forward_plasma = {'input_dim':3,'hidden_dim':30,'hidden_nums':3,'output_dim':28,'block_layer_nums':3}
hyperparas_forward_urine = {'input_dim':3,'hidden_dim':30,'hidden_nums':3,'output_dim':15,'block_layer_nums':3}
learning_rate = 0.001
num_epochs = 300


# 初始化模型、损失函数和优化器
model = ResNN_Reverse(hyperparas_reverse).to(device)
model_forward_plasma = ResNN_Forward(hyperparas_forward_plasma).to(device)
model_forward_plasma.load_state_dict(torch.load('Python\ForwardFitNN\Settled_Model\\threeTo28\\4\model3.pth'))
model_forward_urine = ResNN_Forward(hyperparas_forward_urine).to(device)
model_forward_urine.load_state_dict(torch.load('Python\ForwardFitNN\Settled_Model\\threeTo15\\bps\\model1.pth'))
model_forward_urineg = ResNN_Forward(hyperparas_forward_urine).to(device)
model_forward_urineg.load_state_dict(torch.load('Python\ForwardFitNN\Settled_Model\\threeTo15\\bpsg\\model1.pth'))
loss_function=nn.MSELoss()

def criterion(output,label,input,mode = 1):
    if mode == 1:
        return loss_function(output,label)
    if mode == 2:
        output_forward_plasma = model_forward_plasma(output) #模型输出的三参数代入至PBPK的拟合网络中得到一个代表28个血液药含量采样点的数组
        output_forward_plasma = label_transform_reverse_tensor(output_forward_plasma)
        input_inverse = inverse_transform(input, X_mean, X_std) #将输入特征解除标准化
        return loss_function(output_forward_plasma,input_inverse[:,:28])
    if mode == 3:
        output_forward_urine = model_forward_urine(output) #模型输出的三参数代入至PBPK的拟合网络中得到一个代表15个尿液bps含量采样点的数组
        output_forward_urine = label_transform_reverse_tensor(output_forward_urine)
        input_inverse = inverse_transform(input, X_mean, X_std) #将输入特征解除标准化
        return loss_function(output_forward_urine,input_inverse[:,28:43])
    if mode == 4:
        output_forward_urineg = model_forward_urineg(output) #模型输出的三参数代入至PBPK的拟合网络中得到一个代表15个尿液bpsg含量采样点的数组
        output_forward_urineg = label_transform_reverse_tensor(output_forward_urineg)
        input_inverse = inverse_transform(input, X_mean, X_std) #将输入特征解除标准化
        return loss_function(output_forward_urineg,input_inverse[:,43:68])
    

    if mode == 5:
        output_forward_plasma = model_forward_plasma(output) #模型输出的三参数代入至PBPK的拟合网络中得到一个代表28个血液药含量采样点的数组
        output_forward_plasma = label_transform_reverse_tensor(output_forward_plasma)
        output_forward_urine = model_forward_urine(output) #模型输出的三参数代入至PBPK的拟合网络中得到一个代表15个尿液bps含量采样点的数组
        output_forward_urine = label_transform_reverse_tensor(output_forward_urine)
        output_forward_urineg = model_forward_urineg(output) #模型输出的三参数代入至PBPK的拟合网络中得到一个代表15个尿液bpsg含量采样点的数组
        output_forward_urineg = label_transform_reverse_tensor(output_forward_urineg)
        input_inverse = inverse_transform(input, X_mean, X_std) #将输入特征解除标准化
        loss0 = loss_function(output,label)
        loss1 = loss_function(output_forward_plasma,input_inverse[:,:28])
        loss3 = loss_function(output_forward_urine,input_inverse[:,28:43])
        loss5 = loss_function(output_forward_urineg,input_inverse[:,43:68])
        loss0 = 1E2 *loss0
        loss1 = 1E9 *loss1
        loss3 = 1E6 *loss3
        loss5 = 1E5 *loss5
        loss = loss0 + loss1 + loss3 + loss5
        return loss, loss0, loss1, loss3, loss5




optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 准备 DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=32)

patience_counter = 0
patience_on  = 0
patience = 9
stop_training = 0
# 热键函数
def on_press(key):
    global patience_on
    global stop_training
    if key.name == 's':#终止训练，储存最好模型
        print("Training stopped. Saving current best model...")
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        best_model = ResNN_Reverse(hyperparas_reverse).to(device)
        best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))
        # 保存效果最好的模型
        torch.save(best_model.state_dict(), 'Python\optim\Temporary_Model\model_best.pth')
        stop_training = 1

    if key.name == 'q': #中途储存当前最好模型，但并不终止训练
        print("Saving current best model to pause1...")
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        best_model = ResNN_Reverse(hyperparas_reverse).to(device)
        best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))
        # 保存效果最好的模型
        torch.save(best_model.state_dict(), 'Python\optim\Temporary_Model\model_pause1.pth')
        
    if key.name == 'w': #中途储存当前最好模型，但并不终止训练
        print("Saving current best model to pause2...")
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        best_model = ResNN_Reverse(hyperparas_reverse).to(device)
        best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))
        # 保存效果最好的模型
        torch.save(best_model.state_dict(), 'Python\optim\Temporary_Model\model_pause2.pth')

    if key.name == 'e': #中途储存当前最好模型，但并不终止训练
        print("Saving current best model to pause3...")
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        best_model = ResNN_Reverse(hyperparas_reverse).to(device)
        best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))
        # 保存效果最好的模型
        torch.save(best_model.state_dict(), 'Python\optim\Temporary_Model\model_pause3.pth')

    if key.name == 'o': #开启early stopping
        patience_on  = 1
        print("early stopping is turned on")
        
    
keyboard.on_press(on_press)
# 训练模型
best_val_loss = float('inf')
for epoch in range(num_epochs):
    if stop_training:
        break
    model.train()
    train_loss = 0.0
    loss1_total =0.0
    loss2_total =0.0
    loss3_total =0.0
    loss4_total =0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss,_,_,_,_= criterion(outputs, labels,inputs, mode =5)
        loss.backward()
        optimizer.step()
        train_loss += (loss/len(inputs)).item()

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)

            val_loss_single,loss1,loss2,loss3,loss4 = criterion(val_outputs, val_labels, val_inputs, mode =5)
            loss1 = loss1/len(val_inputs)
            loss2 = loss2/len(val_inputs)
            loss3 = loss3/len(val_inputs)
            loss4 = loss4/len(val_inputs)
            val_loss += val_loss_single/len(val_inputs)
            loss1_total += loss1
            loss2_total += loss2
            loss3_total += loss3
            loss4_total += loss4

    #if val_loss / len(val_loader) <0.002:
        #patience_on = 1
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存效果最好的模型
        torch.save(model.state_dict(), 'Python\optim\Temporary_Model\model_best.pth')
    else:
        if patience_on == 1:
            patience_counter += 1
    
    if patience_counter >= patience:
        print(f'在第{epoch+1}个epoch处，Validation loss did not improve for {patience} epochs. Early stopping...')
        print(f'best validation loss : {best_val_loss/ len(val_loader)}')
        break
    if patience_on == 0:
        print(f'Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, best val-loss now : {best_val_loss / len(val_loader)}')
        print(f'Validation Loss Part: A{loss1_total/ len(val_loader)}, B{loss2_total/ len(val_loader)}, C{loss3_total/ len(val_loader)}, D{loss4_total/ len(val_loader)}')
    else: 
        print(f'Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, earlystopping is on, {patience_counter}steps after last bestloss, best val-loss now : {best_val_loss / len(val_loader)}') 


keyboard.unhook_all()
# 加载效果最好的模型
best_model = ResNN_Reverse(hyperparas_reverse).to(device)
best_model.load_state_dict(torch.load('Python\optim\Temporary_Model\model_best.pth'))

# 在测试集上评估模型
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1,shuffle = None)

test_loss = 0.0
with torch.no_grad():
    count = 0
    for test_inputs, test_labels in test_loader:
        
        test_outputs = best_model(test_inputs)
        if count == 0:
            example_True_3para = test_labels
            example_FromNN_3para = test_outputs
            count = 1
        
        test_loss_single,_,_,_,_ = criterion(test_outputs, test_labels,test_inputs, mode =5)
        test_loss += test_loss_single/len(test_inputs)

        if test_loss_single>0.5 and count == 1:
            example_True_3para = test_inputs
            example_FromNN_3para = test_outputs
            count = 2 
            print(test_loss_single)    
        
    print(f'Test Loss: {test_loss / len(test_loader)}')


#模型在测试集跑出来的三参数计算出来的浓度曲线和真实三参数计算出来的浓度曲线对比
example_True_3para = example_True_3para.cpu().numpy()
print(example_True_3para)

example_FromNN_3para=example_FromNN_3para.cpu().numpy()
print(example_FromNN_3para)

print(f'Test Loss: {test_loss / len(test_loader)}')

id = 0
time = np.arange(0,75,0.005)

plasmaTrue,urineTrue,urinegTrue  =  BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras = example_True_3para ,mode = '63')
plasmaFromNN,urineFromNN,urinegFromNN  =  BPS_BPTK_MultiParas(t = time,volunteer_ID =id, paras = example_FromNN_3para ,mode = '63')


plt.subplot(221)
plt.plot(time,plasmaFromNN[0,:],label = 'FromNN_result')
plt.plot(time,plasmaTrue[0,:],label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()

plt.subplot(222)
plt.plot(time,urineFromNN[0,:],label = 'FromNN_result')
plt.plot(time,urineTrue[0,:],label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()

plt.subplot(223)
plt.plot(time,urinegFromNN[0,:],label = 'FromNN_result')
plt.plot(time,urinegTrue[0,:],label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()

plt.subplot(224)
plt.plot(time,plasmaFromNN[0,:],label = 'FromNN_result')
plt.plot(time,plasmaTrue[0,:],label = 'True_result')
plt.xlabel('time(h)')
plt.ylabel('concentration of BPS in plasma')
plt.legend()

plt.show()

