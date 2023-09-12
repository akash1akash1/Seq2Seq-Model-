# -*- coding: utf-8 -*-
"""M22MA007_Assign2_Q2_DL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/108oNTBrGVATYc8LOcGNUe8urREtKfTdT
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

!unzip /content/drive/MyDrive/DL_Assign/household_power_consumption.zip -d /content

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])

# Drop missing values
df.replace('?', np.nan, inplace=True)
df = df.astype('float32')
df.fillna(method='ffill', inplace=True)

# Select the features
features = [ 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
X = df[features].values

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Select the target
y = df['Global_active_power'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Define the dataset and dataloader
train_dataset = TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1))
test_dataset = TensorDataset(X_test.unsqueeze(1), y_test.unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=72, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=72, shuffle=False)

'''# Define the LSTM model for Consuption_LSTMModel'''
class Consuption_LSTMModel(nn.Module):
    def __init__(self, inp_len, hid_len, out_len):
        super(Consuption_LSTMModel, self).__init__()
        self.hidden_size = hid_len
        self.lstm = nn.LSTM(inp_len, hid_len, batch_first=True)
        self.fc = nn.Linear(hid_len, out_len)
        
    def forward(self, input):
        size_=input.size(0)
        h_conn = torch.zeros(1, size_, self.hidden_size)
        c_connect = torch.zeros(1, size_, self.hidden_size)
        t=(h_conn, c_connect)
        output_, _ = self.lstm(input,t )
        out = self.fc(output_[:, -1, :])
        return out

''' Initialization of the model, loss function, and optimizer'''
inp_len = X.shape[1]
hid_len = 10
out_len = 1
model = Consuption_LSTMModel(inp_len, hid_len, out_len)
criterion = nn.MSELoss()
Adamoptimizer = optim.Adam(model.parameters(), lr=0.001)

'''# Training the model'''
num_epochs=5
loss_arr=[]
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        Adamoptimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        Adamoptimizer.step()
        running_loss += loss.item()
    # print('Epoch: %d, Loss: %f'  (epoch+1, running_loss/len(train_loader)))
    print('Epoch: %d, Loss: %.7f' % (epoch+1, running_loss/len(train_loader)))
    loss_arr.append(running_loss/len(train_loader))

import matplotlib.pyplot as plt

plt.plot(range(num_epochs), loss_arr,label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

'''# Evaluatation of  the model on the test data'''
model.eval()
y_pred = []
y_true = []
total=0
test_loss=0
with torch.no_grad():
    for i,data in enumerate(test_loader):
        inputs, labels = data
        outputs = model(inputs)
        y_pred.append(outputs)
        y_true.append(labels)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        # total += target.size(0)

# test_acc =  correct / total
test_loss /= len(test_loader)

print(f"Test loss: {test_loss:.7f}")

# print(y_pred)

# print(y_true)

from sklearn.metrics import mean_squared_error, r2_score
  # Concatenate the predicted and true values
def plot(y_pred,y_true):
  y_pred_ = torch.cat(y_pred).numpy()
  y_true_ = torch.cat(y_true).numpy()

  # Calculate the mean squared error
  mean_squr_error = mean_squared_error(y_true_, y_pred_)
  # Calculate the R-squared score
  r2_scr = r2_score(y_true_, y_pred_)

  # Print the mean squared error and R-squared score
  print('Mean Squared Error:', mean_squr_error)
  print('R-squared Score:', r2_scr)


  # Plot the actual and predicted global active power for the test data
  plt.plot(y_true_)
  plt.plot(y_pred_)
  plt.legend(['Actual', 'Predicted'])
  plt.show()

plot(y_pred,y_true)

y_pred_ = torch.cat(y_pred).numpy()
y_true_ = torch.cat(y_true).numpy()
y_pred_falten=y_pred_.reshape(1,-1)
y_true_falten=y_true_.reshape(1,-1)
plt.plot(y_true_falten[0][:10],linestyle = 'dotted')
plt.plot(y_pred_falten[0][:10],linestyle = 'dotted')
plt.legend(['Actual', 'Predicted'])
plt.show()

"""**Now Spliiting Data to 70:30 and checking**"""

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Define the dataset and dataloader
train_dataset = TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1))
test_dataset = TensorDataset(X_test.unsqueeze(1), y_test.unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=72, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=72, shuffle=False)

# Define the LSTM model
class Consuption_LSTMModel(nn.Module):
    def __init__(self, inp_len, hid_len, out_len):
        super(Consuption_LSTMModel, self).__init__()
        self.hidden_size = hid_len
        self.lstm = nn.LSTM(inp_len, hid_len, batch_first=True)
        self.fc = nn.Linear(hid_len, out_len)
        
    def forward(self, input):
        size_=input.size(0)
        h_conn = torch.zeros(1, size_, self.hidden_size)
        c_connect = torch.zeros(1, size_, self.hidden_size)
        t=(h_conn, c_connect)
        output_, _ = self.lstm(input,t )
        out = self.fc(output_[:, -1, :])
        return out

# Initialize the model, loss function, and optimizer
inp_len = X.shape[1]
hid_len = 10
out_len = 1
model = Consuption_LSTMModel(inp_len, hid_len, out_len)
criterion = nn.MSELoss()
Adamoptimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs=5
loss_arr=[]
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        Adamoptimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        Adamoptimizer.step()
        running_loss += loss.item()
    # print('Epoch: %d, Loss: %f'  (epoch+1, running_loss/len(train_loader)))
    print('Epoch: %d, Loss: %.7f' % (epoch+1, running_loss/len(train_loader)))
    loss_arr.append(running_loss/len(train_loader))

import matplotlib.pyplot as plt

plt.plot(range(num_epochs), loss_arr,label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test data
model.eval()
y_pred = []
y_true = []
total=0
test_loss=0
with torch.no_grad():
    for i,data in enumerate(test_loader):
        inputs, labels = data
        outputs = model(inputs)
        y_pred.append(outputs)
        y_true.append(labels)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        # total += target.size(0)

# test_acc =  correct / total
test_loss /= len(test_loader)

print(f"Test loss: {test_loss:.7f}")

from sklearn.metrics import mean_squared_error, r2_score
  # Concatenate the predicted and true values
def plot(y_pred,y_true):
  y_pred_ = torch.cat(y_pred).numpy()
  y_true_ = torch.cat(y_true).numpy()

  # Calculate the mean squared error
  mean_squr_error = mean_squared_error(y_true_, y_pred_)
  # Calculate the R-squared score
  r2_scr = r2_score(y_true_, y_pred_)

  # Print the mean squared error and R-squared score
  print('Mean Squared Error:', mean_squr_error)
  print('R-squared Score:', r2_scr)


  # Plot the actual and predicted global active power for the test data
  plt.plot(y_true_)
  plt.plot(y_pred_)
  plt.legend(['Actual', 'Predicted'])
  plt.show()

plot(y_pred,y_true)

y_pred_ = torch.cat(y_pred).numpy()
y_true_ = torch.cat(y_true).numpy()
y_pred_falten=y_pred_.reshape(1,-1)
y_true_falten=y_true_.reshape(1,-1)
plt.plot(y_true_falten[0][:10],linestyle = 'dotted')
plt.plot(y_pred_falten[0][:10],linestyle = 'dotted')
plt.legend(['Actual', 'Predicted'])
plt.show()

print("some sample of y_pred_falten\n",y_pred_falten[0][:10])
print("some sample of y_true_falten\n",y_true_falten[0][:10])
