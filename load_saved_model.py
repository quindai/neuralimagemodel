#Training the model
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Load the model
model = models.load_model('model_image_classifier.h5')
print(model.summary())

# use the model
img = cv.imread('img/bird22.jpg')
img = cv.resize(img, (32, 32))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

prediction = model.predict(np.array([img])/255.0)

print(prediction)
index = np.argmax(prediction)
print(f'Prediction: {class_names[index]}')

plt.imshow(img, cmap=plt.cm.binary)
plt.show()