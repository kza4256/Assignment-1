# Assignment-1
This code is a machine learning project, for image classification using a convolutional neural network (CNN) with TensorFlow and Keras. Let me break down the main components and explain the output plots.

Explanation:

   1. Import Libraries and Setup:

         a. Make sure to install: !pip install --upgrade tensorflow.      
         b. Importing necessary libraries: NumPy, Pandas, and TensorFlow.\
      
         c. It sets up the data directory and loads the dataset using TensorFlow's TPU (Tensor Processing Unit) strategy if available.

  3. Data Preprocessing:
          a. load and data augmentation
          b. Import necessary libraries: tensorflow import image as tf_image
          c. The code defines functions for reading and decoding TFRecord files, which are a common format for storing large datasets.
             It also sets up data augmentation functions for image preprocessing.

  4.  Model Definition: Import tensorflow.keras import layers, models
            A simple CNN model is defined using Keras with convolutional layers, max-pooling layers, and fully connected layers.
            The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.

  5. Transfer Learning:
        Transfer learning is applied by using a pre-trained VGG16 model as a feature extractor. Only the last layer is added for classification.
        The model is compiled again for transfer learning.

  6. Training:
        The model is trained using the training dataset and validated on the validation dataset. The training history is stored.
        The training is done for 5 epochs, and the accuracy and loss are printed for each epoch.

  7. Hyperparameter Tuning:
        The code then explores different learning rates (0.001, 0.01, and 0.1) and trains the model for each learning rate.
        The training and validation accuracy and loss for each learning rate are printed.

  8. Plotting: import matplotlib.pyplot as plt
        Finally, the code uses Matplotlib to plot the training and validation accuracy over epochs for the original training and the learning rate exploration.
