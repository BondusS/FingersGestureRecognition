import pandas as pd
import os
import cv2
import numpy as np
import tensorflow as tf

# Загрузка данных
trainpath = os.listdir("fingers/train")
testpath = os.listdir("fingers/test")
traindata = ["fingers/train/" + i for i in trainpath]
testdata = ["fingers/test/" + i for i in testpath]
traindata = pd.DataFrame(traindata, columns=['Filepath'])
testdata = pd.DataFrame(testdata, columns=['Filepath'])
traindata['Y'] = traindata['Filepath'].apply(lambda a: a[-6:-5])
testdata['Y'] = testdata['Filepath'].apply(lambda a: a[-6:-5])
imagestrain = []
imagestest = []
for path in traindata['Filepath']:
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = image.reshape(128, 128, 3)
    imagestrain.append(image)
for path in testdata['Filepath']:
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = image.reshape(128, 128, 3)
    imagestest.append(image)
ytrain = []
ytest = []
for i in traindata['Y']:
    mas = []
    for j in range(6):
        mas.append(0)
    mas[int(i)] = 1
    ytrain.append(np.array(mas))
for i in testdata['Y']:
    mas = []
    for j in range(6):
        mas.append(0)
    mas[int(i)] = 1
    ytest.append(np.array(mas))
ytrain = np.array(ytrain)
ytest = np.array(ytest)
del traindata, testdata

# Построение модели
model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
                             tf.keras.layers.Conv2D(filters=32,
                                                    kernel_size=3,
                                                    strides=(2, 2),
                                                    padding='same',
                                                    activation='relu',
                                                    input_shape=(128, 128, 3)),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.ReLU(),
                             tf.keras.layers.DepthwiseConv2D(kernel_size=32,
                                                             padding='same',
                                                             activation='relu'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.ReLU(),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(units=6, activation="softmax")])
model.summary()

# Компиляция и обучение модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=tf.stack(imagestrain),
          y=ytrain,
          epochs=1,
          verbose=1,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=2,
                                                      restore_best_weights=True)])

# Показатели модели на тестовом наборе
loss, acc = model.evaluate(x=tf.stack(imagestest),
                           y=ytest)
print('Test loss:', loss, ', test accuracy:', acc)

# Сохранение модели
Model_name = 'model_' + str((acc // 0.0001) * 0.0001)
model.save(Model_name)
