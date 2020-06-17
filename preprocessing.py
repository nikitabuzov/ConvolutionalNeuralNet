import numpy as np
from tqdm import tqdm

# load the data set
train_data = np.load('train_data.npy')
test = np.load('test_data.npy')



for j in tqdm(range(50000)):
    R = train_data[j,:1024]
    G = train_data[j,1024:2048]
    B = train_data[j,2048:3072]
    image = []
    pixel =[]
    for i in range(1024):
        pixel = []
        pixel = np.append(pixel, R[i])
        pixel = np.append(pixel, G[i])
        pixel = np.append(pixel, B[i])
        image = np.append(image, pixel)
    train_data[j] = image

for j in tqdm(range(10000)):
    R = test[j,:1024]
    G = test[j,1024:2048]
    B = test[j,2048:3072]
    image = []
    pixel =[]
    for i in range(1024):
        pixel = []
        pixel = np.append(pixel, R[i])
        pixel = np.append(pixel, G[i])
        pixel = np.append(pixel, B[i])
        image = np.append(image, pixel)
    test[j] = image

np.save('train_data_v2.npy', train_data)
np.save('test_data_v2.npy', test)
