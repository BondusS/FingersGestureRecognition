import pandas as pd
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Загрузка данных
testpath = os.listdir("fingers/test")
testdata = ["fingers/test/" + i for i in testpath]
testdata = pd.DataFrame(testdata, columns=['Filepath'])
testdata['Y'] = testdata['Filepath'].apply(lambda a: a[-6:-5])
imagestest = []
for path in testdata['Filepath']:
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = image.reshape(128, 128, 3)
    imagestest.append(image)
ytest = []
for i in testdata['Y']:
    mas = []
    for j in range(6):
        mas.append(0)
    mas[int(i)] = 1
    ytest.append(np.array(mas))
ytest = np.array(ytest)
del testdata

# Загрузка модели
model = tf.keras.models.load_model('model_0.9933')
model.summary()

# Получение ответа от модели
def GetAnswer(img):
    return np.argmax(model.predict(tf.stack([img]))[0])

# Построение графика с ответами
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(imagestest[i])
    ax.axis('off')
    ax.set_title(f"{GetAnswer(imagestest[i])} fingers")
plt.show()
