import matplotlib.pyplot as plt

from utlis import *
from sklearn.model_selection import train_test_split

path = 'myData'
data = importDataInfo(path)

data = balancedData(data, display=False)

imagesPath, steerings = loadData(path, data)
#print(imagesPath[0], steering[0])

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

model = createModel()
model.summary()

history = model.fit(batchGenerator(xTrain, yTrain, 100, 1), steps_per_epoch = 300, epochs = 10,
          validation_data = batchGenerator(xVal, yVal, 100, 0), validation_steps = 200)

model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()