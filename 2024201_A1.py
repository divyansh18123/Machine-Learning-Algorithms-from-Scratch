import idx2numpy
import numpy as np
from numpy import random
import math

import sklearn
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd


with open('archive 2/train-images.idx3-ubyte', 'rb') as f:
    whole_train_images_arr = idx2numpy.convert_from_file(f)

with open('archive 2/train-labels.idx1-ubyte', 'rb') as f:
    whole_train_labels_arr = idx2numpy.convert_from_file(f)

with open('archive 2/t10k-images.idx3-ubyte', 'rb') as f:
    whole_test_images_arr = idx2numpy.convert_from_file(f)

with open('archive 2/t10k-labels.idx1-ubyte', 'rb') as f:
    whole_test_labels_arr = idx2numpy.convert_from_file(f)


whole_train_image_0 = []
whole_train_image_2 = []
whole_train_image_1 = []


for i in range(len(whole_train_labels_arr)):
    if whole_train_labels_arr[i] == 0:
        whole_train_image_0.append(whole_train_images_arr[i])
    elif whole_train_labels_arr[i] == 1:
        whole_train_image_1.append(whole_train_images_arr[i])
    elif whole_train_labels_arr[i] == 2:
        whole_train_image_2.append(whole_train_images_arr[i])


whole_train_image_0 = np.array(whole_train_image_0)
whole_train_image_1 = np.array(whole_train_image_1)
whole_train_image_2 = np.array(whole_train_image_2)

train_0_random_100 = np.array(random.choice(len(whole_train_image_0), size = (100), replace = False))
train_1_random_100 = np.array(random.choice(len(whole_train_image_1), size = (100), replace = False))
train_2_random_100 = np.array(random.choice(len(whole_train_image_2), size = (100), replace = False))

final_train_0 = whole_train_image_0[train_0_random_100]
final_train_1 = whole_train_image_1[train_1_random_100]
final_train_2 = whole_train_image_2[train_2_random_100]




whole_test_image_0 = []
whole_test_image_2 = []
whole_test_image_1 = []


for i in range(len(whole_test_labels_arr)):
    if whole_test_labels_arr[i] == 0:
        whole_test_image_0.append(whole_test_images_arr[i])
    elif whole_test_labels_arr[i] == 1:
        whole_test_image_1.append(whole_test_images_arr[i])
    elif whole_test_labels_arr[i] == 2:
        whole_test_image_2.append(whole_test_images_arr[i])


whole_test_image_0 = np.array(whole_test_image_0)
whole_test_image_1 = np.array(whole_test_image_1)
whole_test_image_2 = np.array(whole_test_image_2)

test_0_random_100 = np.array(random.choice(len(whole_test_image_0), size = (100), replace = False))
test_1_random_100 = np.array(random.choice(len(whole_test_image_1), size = (100), replace = False))
test_2_random_100 = np.array(random.choice(len(whole_test_image_2), size = (100), replace = False))

final_test_0 = whole_test_image_0[test_0_random_100]
final_test_1 = whole_test_image_1[test_1_random_100]
final_test_2 = whole_test_image_2[test_2_random_100]


# Normalize

final_train_0 = final_train_0.reshape(len(final_train_0), -1)
final_train_1 = final_train_1.reshape(len(final_train_1), -1)
final_train_2 = final_train_2.reshape(len(final_train_2), -1)

final_test_0 = final_test_0.reshape(len(final_test_0), -1)
final_test_1 = final_test_1.reshape(len(final_test_1), -1)
final_test_2 = final_test_2.reshape(len(final_test_2), -1)


final_train_0 = final_train_0/255.0
final_train_1 = final_train_1/255.0
final_train_2 = final_train_2/255.0


final_test_0 = final_test_0/255.0
final_test_1 = final_test_1/255.0
final_test_2 = final_test_2/255.0

# MLE
sum = np.zeros((784,))
for i in final_train_0:
    sum = sum + i

mu_0 = sum/100

sum = np.zeros((784,))
for i in final_train_1:
    sum = sum + i

mu_1 = sum/100

sum = np.zeros((784,))
for i in final_train_2:
    sum = sum + i

mu_2 = sum/100


sum = np.zeros((784,784))
for i in final_train_0:
    sum += np.matmul(((i - mu_0).reshape(1,-1)).T, (i - mu_0).reshape(1,-1))

cov_0 = sum/100

sum = np.zeros((784,784))
for i in final_train_1:
    sum += np.matmul(((i - mu_1).reshape(1,-1)).T, (i - mu_1).reshape(1,-1))

cov_1 = sum/100

sum = np.zeros((784,784))
for i in final_train_2:
    sum += np.matmul(((i - mu_2).reshape(1,-1)).T, (i - mu_2).reshape(1,-1))

cov_2 = sum/100


mu = (mu_0 + mu_1 + mu_2)/3

sum = np.zeros((784,784))
for i in final_train_0:
    sum += np.matmul(((i - mu).reshape(1,-1)).T, (i - mu).reshape(1,-1))
for i in final_train_1:
    sum += np.matmul(((i - mu).reshape(1,-1)).T, (i - mu).reshape(1,-1))
for i in final_train_2:
    sum += np.matmul(((i - mu).reshape(1,-1)).T, (i - mu).reshape(1,-1))

cov = sum/300

reg_cov_0 = cov_0 + 0.001 * np.identity(784)
reg_cov_1 = cov_1 + 0.001 * np.identity(784)
reg_cov_2 = cov_2 + 0.001 * np.identity(784)
reg_cov = cov + 0.001 * np.identity(784)




# LDA QDA
def lda_func(mu_0, mu_1, mu_2, cov, x):
    
    try:
        np.linalg.inv(cov)
    except:
        print("Covariance matrix is not invertible")
        return
    inv = np.linalg.inv(cov)
    g0x = np.matmul(np.matmul(x, inv), mu_0.reshape(1,-1).T) -(1/2) * np.matmul(np.matmul(mu_0, inv), mu_0.reshape(1,-1).T)
    g1x = np.matmul(np.matmul(x, inv), mu_1.reshape(1,-1).T) -(1/2) * np.matmul(np.matmul(mu_1, inv), mu_1.reshape(1,-1).T)
    g2x = np.matmul(np.matmul(x, inv), mu_2.reshape(1,-1).T) -(1/2) * np.matmul(np.matmul(mu_2, inv), mu_2.reshape(1,-1).T)

    if max(g0x, g1x, g2x) == g0x:
        return 0, g0x.tolist()[0], g1x.tolist()[0], g2x.tolist()[0]
    elif max(g0x, g1x, g2x) == g1x:
        return 1, g0x.tolist()[0], g1x.tolist()[0], g2x.tolist()[0]
    elif max(g0x, g1x, g2x) == g2x:
        return 2, g0x.tolist()[0], g1x.tolist()[0], g2x.tolist()[0]

def qda_func(mu_0, mu_1, mu_2, cov_0, cov_1, cov_2, x):
    try:
        np.linalg.inv(cov_0)
        np.linalg.inv(cov_1)
        np.linalg.inv(cov_2)
    except:
        print("Covariance matrix is not invertible")
        return
    

    g0x = -(1/2)*np.linalg.slogdet(cov_0)[1] -(1/2)* np.matmul(np.matmul((x - mu_0), np.linalg.inv(cov_0)), (x - mu_0).reshape(1,-1).T)
    g1x = -(1/2)*np.linalg.slogdet(cov_1)[1] -(1/2)* np.matmul(np.matmul((x - mu_1), np.linalg.inv(cov_1)), (x - mu_1).reshape(1,-1).T)
    g2x = -(1/2)*np.linalg.slogdet(cov_2)[1] -(1/2)* np.matmul(np.matmul((x - mu_2), np.linalg.inv(cov_2)), (x - mu_2).reshape(1,-1).T)

    if max(g0x, g1x, g2x) == g0x:
        return 0, g0x.tolist()[0], g1x.tolist()[0], g2x.tolist()[0]
    elif max(g0x, g1x, g2x) == g1x:
        return 1, g0x.tolist()[0], g1x.tolist()[0], g2x.tolist()[0]
    elif max(g0x, g1x, g2x) == g2x:
        return 2, g0x.tolist()[0], g1x.tolist()[0], g2x.tolist()[0]


correct_lda_0 = 0
correct_lda_1 = 0
correct_lda_2 = 0
correct_qda_0 = 0
correct_qda_1 = 0
correct_qda_2 = 0


for i in final_test_0:
    predicted, _, _, _ = lda_func(mu_0, mu_1, mu_2, reg_cov, i)

    if predicted == 0:
        correct_lda_0 += 1

    predicted, _, _, _ = qda_func(mu_0, mu_1, mu_2, reg_cov_0, reg_cov_1, reg_cov_2, i)

    if predicted == 0:
        correct_qda_0 += 1


for i in final_test_1:
    predicted, _, _, _ = lda_func(mu_0, mu_1, mu_2, reg_cov, i)

    if predicted == 1:
        correct_lda_1 += 1

    predicted, _, _, _ = qda_func(mu_0, mu_1, mu_2, reg_cov_0, reg_cov_1, reg_cov_2, i)

    if predicted == 1:
        correct_qda_1 += 1

for i in final_test_2:
    predicted, _, _, _ = lda_func(mu_0, mu_1, mu_2, reg_cov, i)

    if predicted == 2:
        correct_lda_2 += 1

    predicted, _, _, _ = qda_func(mu_0, mu_1, mu_2, reg_cov_0, reg_cov_1, reg_cov_2, i)

    if predicted == 2:
        correct_qda_2 += 1
    

print("LDA Accuracy for 0 = ", correct_lda_0/100)
print("LDA Accuracy for 1 = ", correct_lda_1/100)
print("LDA Accuracy for 2 = ", correct_lda_2/100)
print("QDA Accuracy for 0 = ", correct_qda_0/100)
print("QDA Accuracy for 1 = ", correct_qda_1/100)
print("QDA Accuracy for 2 = ", correct_qda_2/100)
print("LDA Accuracy = ", (correct_lda_0 + correct_lda_1 + correct_lda_2)/300)
print("QDA Accuracy = ", (correct_qda_0 + correct_qda_1 + correct_qda_2)/300)

if (correct_lda_0 + correct_lda_1 + correct_lda_2)/300 >= (correct_qda_0 + correct_qda_1 + correct_qda_2)/300:
    print("LDA performed better")
else:
    print("QDA performed better")

j = random.randint(100)
i = final_test_0[j]
print("Discriminant values of random test point from 0 index", j)
print("Prediction, g0x, g1x, g2x - ", lda_func(mu_0, mu_1, mu_2, reg_cov, i))
print("Prediction, g0x, g1x, g2x - ", qda_func(mu_0, mu_1, mu_2, reg_cov_0, reg_cov_1, reg_cov_2, i))

j = random.randint(100)
i = final_test_1[j]
print("Discriminant values of random test point from 1, index", j)
print("Prediction, g0x, g1x, g2x - ", lda_func(mu_0, mu_1, mu_2, reg_cov, i))
print("Prediction, g0x, g1x, g2x - ", qda_func(mu_0, mu_1, mu_2, reg_cov_0, reg_cov_1, reg_cov_2, i))

j = random.randint(100)
i = final_test_2[j]
print("Discriminant values of random test point from 2, index", j)
print("Prediction, g0x, g1x, g2x - ", lda_func(mu_0, mu_1, mu_2, reg_cov, i))
print("Prediction, g0x, g1x, g2x - ", qda_func(mu_0, mu_1, mu_2, reg_cov_0, reg_cov_1, reg_cov_2, i))



# TSNE
model = sklearn.manifold.TSNE(n_components = 2, random_state = 42)
label_0 = np.full((100, 1), 0)
label_1 = np.full((100, 1), 1)
label_2 = np.full((100, 1), 2)

labels = np.append(np.append(label_0, label_1, axis = 0), label_2, axis = 0)
train_data = np.append(np.append(final_train_0, final_train_1, axis = 0), final_train_2, axis = 0)

tsne_data = model.fit_transform(train_data)
tsne_data = np.append(tsne_data, labels, axis = 1)
df = pd.DataFrame(tsne_data)

df.columns = ["x", "y", "label"]

plt.subplot(1, 2, 1)
sn.scatterplot(data = df, x = "x", y = "y", hue = "label")
plt.title("Train set")




test_data = np.append(np.append(final_test_0, final_test_1, axis = 0), final_test_2, axis = 0)

tsne_data = model.fit_transform(test_data)
tsne_data = np.append(tsne_data, labels, axis = 1)
df = pd.DataFrame(tsne_data)

df.columns = ["x", "y", "label"]

plt.subplot(1,2,2)
sn.scatterplot(data = df, x = "x", y = "y", hue = "label")
plt.title("Test Set")

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.show()