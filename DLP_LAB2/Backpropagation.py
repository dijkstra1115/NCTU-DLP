#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np
import math
import matplotlib.pyplot as plt


# In[103]:


def generate_linear(n = 100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


# In[104]:


def generate_XOR_easy(n = 101):
    inputs = []
    labels = []
    for i in range(n//2+1):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


# In[105]:


def relu(x):
    for i in range(x.shape[1]):
        if x[0, i] > 0:
            continue
        else:
            x[0, i] = 0
    return x


# In[106]:


def der_relu(x):
    for i in range(x.shape[1]):
        if x[0, i] > 0:
            x[0, i] = 1
        else:
            x[0, i] = 0
    return x


# In[107]:


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# In[108]:


def der_sigmoid(x):
    return x * (1.0 - x)


# In[109]:


mode = "XOR"
if mode == "linear":
    inputs, labels = generate_linear()
else:
    inputs, labels = generate_XOR_easy()


# In[110]:


hidden_nums = [10, 10]
w1 = np.random.rand(2, hidden_nums[0])
w2 = np.random.rand(hidden_nums[0], hidden_nums[1])
w3 = np.random.rand(hidden_nums[1], 1)
z1 = np.zeros((1, hidden_nums[0]))
z2 = np.zeros((1, hidden_nums[1]))
learning_rate = 0.1


# In[111]:


plt.title("XOR Learning Curve")
loss_curve = []
epochs = []
for epoch in range(2000):
    total_loss = 0
    for i, x in enumerate(inputs):
        a0 = np.array(x).reshape((1,2)) #要記得都轉換成numpy的型態
        y = np.array(labels[i]).reshape((1,1))
        #forward pass
        z1 = a0 @ w1
        a1 = sigmoid(z1)
        z2 = a1 @ w2
        a2 = sigmoid(z2)
        z3 = a2 @ w3
        a3 = sigmoid(z3)
        predict = a3
        
        #backward pass
        C_z3 = der_sigmoid(a3) * -(y/(predict)-(1-y)/(1-predict))
        C_a2 = (C_z3 @ w3.T)
        w3_gradient = (a2.T @ C_z3)
        w3 = w3 - learning_rate * w3_gradient
        
        C_z2 = der_sigmoid(a2) * C_a2
        C_a1 = (C_z2 @ w2.T)
        w2_gradient = (a1.T @ C_z2)
        w2 = w2 - learning_rate * w2_gradient

        C_z1 = der_sigmoid(a1) * C_a1
        C_a0 =  (C_z1 @ w1.T)
        w1_gradient = (a0.T @ C_z1) #Gradient是要乘input,而不是weight
        w1 = w1 - learning_rate * w1_gradient
        loss = -(y * math.log(predict) + (1-y) * math.log(1-predict))
        total_loss += loss
    if epoch % 100 == 0:
        loss_curve.append(total_loss.item())
        epochs.append(epoch)
        print(f"epoch:{epoch} total_loss:{total_loss.item()}")

plt.plot(epochs,loss_curve)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


# In[112]:


n = 21
if mode == "linear":
    test, test_labels = generate_linear(n)
else:
    test, test_labels = generate_XOR_easy(n)
predict_labels = []


# In[113]:


#testing
for i, x in enumerate(test):
    a0 = np.array(x).reshape((1,2))
    z1 = sigmoid(a0 @ w1)
    z2 = sigmoid(z1 @ w2)
    predict = sigmoid(z2 @ w3)
    print(predict)
    if predict > 0.5:
        predict_labels.append(1)
    else:
        predict_labels.append(0)

predict_labels = np.array(predict_labels).reshape(n, 1)


# In[114]:


print(f"Error rate: {np.sum(np.power(predict_labels - test_labels, 2)) / n}%")


# In[115]:


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()


# In[116]:


show_result(test, test_labels, predict_labels)


# In[ ]:




