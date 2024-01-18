# Neural Network Tuning

Neural networks are at the heart of many machine learning systems, and Espressif makes it clear in their demos. Being able to push a neural network model to the ESTp-EYE is important, but it is also important to be able to understand the structure of the model, and be able to create your own and fine tune it. In this exercise, you will load a well-known dataset, and will use tensorflow to train the model, then fine tune it.

The Cifar-10 dataset is a large collection of images (60,000 images, all in color, of size 32x32), that is typically used to train a classifier (to recognize 10 different classes of images). You can find it [here](https://www.cs.toronto.edu/~kriz/cifar.html). It is a subset of cifar-100, that contains ten times more images, (and cifar-100 is a subset of the full dataset, of 80 million images).

Cifar-10 is so well-known, that many frameworks integrate it directly it is the case for tensorflow. So the first step is to start a new Jupyter notebook, in your tensorflow environment, then load the libraris we need, along with the dataset. As you may remember, tensorflow uses Keras as a wrapper to define the structure of the network. We also load the to_categorical library, which allows us to create categories as numbers (for example, an object of category 3 in a system with 4 categories is automatically represented by this library as [0, 0, 1, 0]):

```shell
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

```

Once we have our data, we can load the data. We tell the engine that there will be a training part, and a test part. For each of them, there will be the image data (x), and the class (y). Now, images typically come as a number between 0 and 255 (if you do not remmember why, go back to our previous labs where we looked at gray images, and concluded that each image ended up being a series of scores, that represent the shade of each zone of the image). In order to be able to work with the images, we normalize the number, i.e. divide the [0,255] score by 255, to get a score between 0 and 1. Then we use the one-hot-encode technique to put each class into a row format as show above.

```shell
# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-hot encode

```

If you are not sure about any part of the code, you can run the commands one line at a time, and call the variables to look at them and better understand their format:

```shell
y_train

```

Once we have our data in memory, we split into train and validation set. We use the first 40K entries for the training, and all entries above 40K (i.e., the last 20K) for the validation:


```shell
# Split training data into train and validation sets
x_train, x_val = x_train[:40000], x_train[40000:]
y_train, y_val = y_train[:40000], y_train[40000:]

```

The next step is to build the neural network. We want to use a CNN (because we work on images and CNNs are efficient for this type of data). As each image is 32x32, the input is going to be 32x32, in RGB (so one score [0,255], normalized to [0,1], for each of the 3 tones). We will then inject them into a first convolutional layer (with a ReLu activation function to eliminate negative scores) that will look at blocks of 3x3 pixesl, the layer has 32 units.
Then, we will inject the otuput into a pooling layer, that reduces the score to 2x2 blocks, thus making it smaller (as you may remember, we care to know if the feature of a target image, for example an X or a O, is here or not, we do not care where the feature is, so concluding on the rpesence of a feature in a zone is sufficient). We repeat the process, injecting he result into another convolutional layer, this time of 64 units (but still 3x3), with a ReLu activation function, then a max pooling layer, of 2x2 again. 

Then we convert our scores for each feature into a single vector (we 'flatten'). Finally, we inject this score vector for each image into a last convolutional layer, with 64 units and a ReLu activation function, and ask the system to output a class score with 10 classes:


```shell
# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

Why did we choose this structure? That's an excellent question! And that is also one goal of this lab. Someone thought about the structure of the dataset, tried a few things, and concluded that this structure would work. But you cannot make this conclusion without trying a few things. So let's first compile the model, then we'll try to refine it. We use the [adam optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) (Adaptive Moment Estimation). It is a comon optimizer for neural entworks, because it can pick variable learning rates (i.e., slow down as it gets closer to 0, so as not to miss the lowest point), and also uses two techniques for the error estimation, to be fast while avoiding the local minima. You can learn about its arguments in Keras [here](https://keras.io/api/optimizers/adam/).

Once we heve defined how the model should be compiled (i.e. what can of equation or algorithm we use in the training), we just call our usual function 'fit' to train it:


```shell
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

```

And as usual.... that's it. Depending on your machine, it may have taken a few minutes to run. Now it is time to see how good the prediction is (again, we kept some data for this type of validation, so we can measure how coherent our model is with the dataset we use).

One great command with Keras in tensorflow is 'evaluate', which, as its name indicates, can give you performance scores on your model. We get the loss, which is the cost of the final model, and its accuracy. Remember, accuracy is supposed to be high if your model works. In general [disclaimer: this depends on the case at hand yada yada] you do not wan an acuracy below 80% to 90%. 80% is low, anything below 70% coudl well be random. You also do not want something too close to 100% in real life (except if it is a training exercise like this one), because it would mean that your model is very biased: as soon as you get new data, your score will drop.

We then plot the accuracy we got at each stage of the training (accuracy should improve with each epoch), and the accuracy we get on each segment of the validation dataset.

```shell
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# Plotting training and validation accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

```

If your code is like mine, performances may be only average. Let us try to improve things. 

  # Adjusting the learning rate

One thing you can try to play with is the learning rate. This is how you would change it. Try a few values (0.0001 is just an example, you can try to add or remove zeros, change the numbers etc.), retrain and plot, and see if you can find a value that gives a better result:

```shell
# Example: Adjusting the learning rate
new_lr = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=new_lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Retrain the model with the new learning rate
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

```

  ## Adjusting the number of epochs

Apart from the learning rate, you can also change the number of epochs. In each epoch, the systm picks a number of images for its training. Changing that number (called the bacth size) can also help. If you have more epochs, you spend more time training, so your learning is more accuract (but at the cost of training time). If you use more images in each batch, you have more comparison points, so you may need less epochs. It is worht trying a few combinations. Here are two examples:

```shell
# Experiment with different batch sizes and epochs
history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_val, y_val))
history = model.fit(x_train, y_train, batch_size=64, epochs=15, validation_data=(x_val, y_val))


```

Try these two anda fe others, then re-compute the performances. Can you find something better than the defaults?

 ## Using a different optimizer

We used the adam optimizer. It is a classic, but not the only game in town, and the [Keras website](https://keras.io/api/optimizers/) gives you the most common ones. If you want to work with AIML, it is a good idea to read a bit about them, you should speak optimizers as well as you speak English. We can try two other very efficient optimzers. To use something else than adam, replace the model.compile section of your code. For example:

```shell
# Using SGD optimizer
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

```
Also:

```shell
# Using RMSprop optimizer
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

```


  ## Using a different activation function

We used a convolution layer witha  ReLu, but sometimes it is more efficient to insert a different activation function, like tanh. You would not replace all the ReLus with tanha, but you could replace one of them. I would look like this:


```shell
# Using a different activation function (e.g., tanh)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='tanh', input_shape=(32, 32, 3)),
    # ... other layers ...
])


```

  ## Using a different network architecture

Changing the architecture of the network, like the number of layers, the number of neurons in each layer, etc., can have a significant impact. Try a few variations. With the learining rate and epoch/batch, this is my goto technique. For example (but try a few other variations and see if you can find somethign that works better):

```shell
# Adding more layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # ... other layers ...
])

# Changing the number of neurons
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    # ... other layers ...
])


```


  
  ## Data augmentation

Data augmentation can significantly improve the performance of the model, especially in image-related tasks. With this technique, you create variations of your images (with added random noise), and the model trains on the real and the added images. In most cases, this technique helps the model be more efficient in recognizing images in the test set, because it was trained on more images, and on images that were not very good (noisy):

```shell
# Adding more layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # ... other layers ...
])

# Changing the number of neurons
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    # ... other layers ...
])


```
  
  ## Using a pre-trained model

Another commonly used approach is to use a pre-trained model. This is a CNN that other people have trained, and that you retrain on your data. One advantage of this technique is that the model you reuse may have been trained on a very large set of images, so it may have learned a large set of features (from a large set of images). Here, you use a technique called transfer learning, where you then used this trained model on your data, essentially asking it ti apply its knowledge of features, to recognize features in your dataset. This is often faster than training on a larger dataset (ore training with many more epochs):

```shell
# Example: Using a pre-trained model (VGG16)
base_model = tf.keras.applications.VGG16(input_shape=(32, 32, 3),
                                         include_top=False,
                                         weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Create a new model on top
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))


```

Which technique, which combination gave you the best results?

## Using an automatic tuner

Now, it is true that there are some tools to help you make that optimizaiton. This manual search you ran, a tool like keras tuner that can run the same type of search automatically. I give you this tool last, because using it is the difference between fixing your car (above) and having someone fixe your car (keras tuner): in this second case, you haven't learned anything about the internals of your learning machine, you just learned to copy/paste (but you knew how to do that before starting the exercise).


```shell
!pip install -U keras-tuner
import kerastuner as kt

```

Then, once you have loaded tensorflow, the data, defined x_train, x_test and y_train, y_test, you can define the model building function:



```shell
def build_model(hp):
    model = tf.keras.models.Sequential()

    # Tune the number of filters in the first two convolutional layers
    for i in range(hp.Int('conv_layers', 1, 3)):
        model.add(tf.keras.layers.Conv2D(filters=hp.Int('filters_' + str(i), min_value=32, max_value=128, step=32),
                                         kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))

    model.add(tf.keras.layers.Flatten())

    # Tune the number of units in the first Dense layer
    model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=128, step=32), activation='relu'))

    # Output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
```

Then, you launch keras tuner from this series of bricks to find the best combination of these bricks (the best structure, the best optimizers, the best learning rate, epoch, bacth size):


```shell
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='cifar10_tuning')

# Early stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Start the search
tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
```


Then you just have to train the best model that keras tuner found:

```shell
# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the model
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)
```

You can then use the plotting functions above to see if the model suggested by the tuner worked better than the defaults. In many cases, it does work better, but in many cases as well, you will end up still tuning it manually.




