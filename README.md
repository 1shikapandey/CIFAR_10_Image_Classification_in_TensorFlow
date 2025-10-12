# CIFAR-10 Image Classification using TensorFlow

A deep learning project that classifies images from the **CIFAR-10 dataset** into **10 categories** using a **Convolutional Neural Network (CNN)** built with **TensorFlow and Keras**.

---

## Overview

This project demonstrates the full workflow of building an image classification model — from data preprocessing to training, evaluation, and visualization.

The **CIFAR-10** dataset contains **60,000 RGB images** of size **32×32 pixels** across **10 classes**:
✈️ Airplane · 🚗 Automobile · 🐦 Bird · 🐱 Cat · 🦌 Deer · 🐶 Dog · 🐸 Frog · 🐴 Horse · 🚢 Ship · 🚛 Truck

---

## Key Features

* End-to-end image classification pipeline
* CNN with **Batch Normalization**, **Dropout**, and **MaxPooling** layers
* **Data Augmentation** for improved generalization
* Training visualization using **Matplotlib**
* Model prediction and accuracy comparison
* Model export as `.h5` file

---

## Tech Stack

| Component     | Description        |
| ------------- | ------------------ |
| **Language**  | Python 3           |
| **Framework** | TensorFlow / Keras |
| **Libraries** | NumPy, Matplotlib  |
| **Dataset**   | CIFAR-10           |

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/CIFAR10-Image-Classification.git
   cd CIFAR10-Image-Classification
   ```

2. Install dependencies:

   ```bash
   pip install tensorflow numpy matplotlib
   ```

3. (Optional) For GPU acceleration:

   ```bash
   pip install tensorflow-gpu
   ```

---

## Dataset Details

The **CIFAR-10** dataset is automatically loaded from TensorFlow’s built-in datasets module.

```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

* **Training images:** 50,000
* **Testing images:** 10,000
* **Image dimensions:** 32 × 32 × 3 (RGB)

All pixel values are normalized to the range **[0, 1]** for faster and more stable training.

---

## Model Architecture

A **Convolutional Neural Network (CNN)** was built using TensorFlow’s functional API.

```
Input (32x32x3)
│
├── Conv2D(32) + BatchNorm + Conv2D(32) + BatchNorm + MaxPool
├── Conv2D(64) + BatchNorm + Conv2D(64) + BatchNorm + MaxPool
├── Conv2D(128) + BatchNorm + Conv2D(128) + BatchNorm + MaxPool
│
├── Flatten
├── Dropout(0.2)
├── Dense(1024, relu)
├── Dropout(0.2)
└── Dense(10, softmax)
```

---

## Training the Model

### Compile

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Train

```python
r = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50
)
```

### Data Augmentation

To enhance performance and prevent overfitting:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = data_gen.flow(x_train, y_train, batch_size=32)

r = model.fit(
    train_gen,
    validation_data=(x_test, y_test),
    epochs=50
)
```

---

## Performance Visualization

Visualize training and validation accuracy:

```python
plt.plot(r.history['accuracy'], label='Training Accuracy', color='red')
plt.plot(r.history['val_accuracy'], label='Validation Accuracy', color='green')
plt.legend()
plt.show()
```

---

## Testing and Prediction

```python
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

image_number = 0
plt.imshow(x_test[image_number])

p = x_test[image_number].reshape(1, 32, 32, 3)
predicted = labels[model.predict(p).argmax()]
original = labels[y_test[image_number][0]]

print(f"Original: {original}, Predicted: {predicted}")
```

Example Output:

```
Original: cat, Predicted: cat
```

---

## Save the Trained Model

```python
model.save('cifar10_cnn_model.h5')
```

This saves your trained model for future use or deployment.

---

## What You’ll Learn

* Fundamentals of CNNs for image classification
* Image preprocessing and normalization techniques
* Building deep learning models using the Keras functional API
* Applying data augmentation for better generalization
* Evaluating model accuracy and visualizing training progress

---

## Sample Visualization

Visualizing a few CIFAR-10 images from the training dataset:

```python
fig, ax = plt.subplots(5, 5)
k = 0
for i in range(5):
    for j in range(5):
        ax[i][j].imshow(x_train[k])
        k += 1
plt.show()
```

---

## Summary

This project showcases how a well-structured CNN can effectively classify small images into multiple categories.
By combining **Batch Normalization**, **Dropout**, and **Data Augmentation**, it achieves a balance between **accuracy** and **generalization**, making it a great foundation for deeper architectures like **ResNet** or **VGG** in future experiments.
