# Logistic Regression on ESP-EYE

Let's continue our exploration of ESP-EYE AIML, this time with logistic regression based on speech recognition. This example uses the espressif esp-nn implementation. Start by cloning their repository from your root directory (e.g., ~/IMT/esp-idf/):

```shell
$ git clone --recursive https://github.com/espressif/esp-tflite-micro.git
```

In the folder, and example sub-folder, you will see 3 examples. Navigate to micro-speech:

```shell
$ cd esp-tflite-micro/examples/micro_speech/ 
```

Let's proceed in two steps: try it, then dissect how it is working. The README.md file in the folder (also [here](https://github.com/espressif/esp-tflite-micro/tree/master/examples/micro_speech)) provides an intro that you may want to read for curiosity sake. 
In general, the goal of the tool is to recognize when you say 'yes' or 'no'. Obviously, the tool relies on the fact that the ESP-EYE has a microphone. So, for each segment of audio captured by the microphone, the system predicts if it contains "yes', 'no', or anything else. It is a basic example, 
but think about its usage in the real world. Although it uses a neural network for a classification (3 categories) task, it illustrates how you can combine the camera with the microphone to give instructions ("open the door", "hey, it's me" in addition to you face, etc.)

Now that you understand the intent, set your ESP-EYE as the target:

```shell
$ idf.py set-target esp32
```

Then build the executables:

```shell
$ idf.py build
```

You can then deploy the executables to your ESP-EYE, and start testing it:

```shell
$ idf.py flash monitor
```

Start talking near the ESP-EYE, each time the tool will detect a 'yes' or a 'no', it should display it on the terminal:

```shell
Heard yes 0.968742 at 07:32
```

Now that you see the general principles, let's look a bit more in details on how the tool works. Espressif here merely implements on the ESP-EYE a tensorflow lite pre-defined example. You can find the example, and its explanation, [here](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech).
Spend time reading that page, and take some notes, for example at the beginning of the next Jupyter notebook, that we will use in the second part of this lab. There are terms that you may not be familiar with (possibly int8/float32, mel spectrogram and a few others), 
but the page will give you a good explanation on what happens when you speak in the camera. The explanation page is important in several ways:

* It provides introductory tools you will need to (answer questions and) understand what happens when we build a similar tool with edge impulse
* It will help you better understand and expand lab 5 on visual object recognition, allowing you to train, build and deploy your own audio or visual model onto ESP-EYE

So do spend time reading the page before moving forward.

# Tensorflow vs. Scikit Learn in Jupyter, and embedded systems

One element you should have retained from the explanation page above is that there is a phase where the model is trained (okay, this is something you know we can do in Jupyter), then deployed to ESP-EYE. This second part may be less clear, and it may also be less clear why 
we use tflite instead of Scikit Learn (or why not both?). Let's shift our focus to this part. Go back to Anaconda navigator, re-launch Jupyter lab if it was closed, and start a new notebook (as usual, give it a name that will help you remember what it was for).

In this next part, we will work a bit with Scikit Learn, then will move to tensorflow. One of the outcomes is to compare both libraries, in terms of command structure, but also efficiency. Our first step is therefore to build a toy model for our comparison. The [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) is a classic, where 3 variants of the iris flower are compared, and it turns out that you can predict which variant a flower is, by looking at its petal and sepal length and width. So the dataset includes the data of 150 flowers of each of the 3 types (with sepal/petal lentgh/width), and each flower variant name is set. This is great for classificaiton, where you can train a model based on x% of the dataset, and then see if you can correctly predict the variant type using the remainder (test) set data. Let's see how we can build this case in Jupyter (and Scikit learn to start).

First, you need as usual a few libraries: numpy for number manipulation, pyplot (from matplotlib) to plot things we find, and seaborn, another library that is very useful to plot confusion matrices and heatmaps:

```shell
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

The Iris dataset is found in many places online. In fact, it is such a classic that most general libraries have it too. So we can pull it directly from scikit learn (you may be interested in researching what other datasets are available). We will also pull from the scikit learn library other tools, like the command to split the data into training/test and the logistic regression tool. These should not be new to you. We will also use a tool to help us scale our numbers (standard scaler, as the petal and sepal numbers are not at the same scale), and a tool to score our prediction performance. In other words, we'll ask scikit learn to test our prediction on the test set, and tell us its score (correct prediction y% of the time). To help us better understand this output, it is often useful to graph the performances against each categories. Each time scikit learn predicts that this test sample is 'category 1', how many times is-it really category 1? How many times is-it category 2? How many times category 3? Plotting the score this way is useful, because it allows us to find which categories are usually easy to predict, and which ones our model tends to get confused between. Unsurprisingly, such graph is called a confusion matrix. 

```shell
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
```

Once we have our tools, let's get going. The first step is to load the iris dataset. It is already nicely organized in two arrays, one is called 'data' and includes the dimensions, and the other one is called 'target' and tells us the variant number. So we can call X the data, and y the target (note how we use capital X for an array that is nxm, where n and m are larger than 1, and lower case for a simple vector (nx1)):

```shell
# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
```
Invest a second to make sure you understand that this makes sense, by looking at the iris set directly:

```shell
iris
```

You should see both arrays and their dimensions.

You may notice that not all the numbers in X seem to be at the same scale. So let's use the Standard Scaler tool to convert each data type (length/width for sepal/petal) and just represent how each sample varies away from that mean. It's a classic trick on AIML when you deal with data that are of different types (and thus different scales):

```shell
# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The steps above were just a preparation, and we would do the same thing with tensorflow, pytorch or any other library. The next step should also be familiar to you by now, we split the data into a training and a test sets. Note that we work from X_scaled, not X (we worked hard to scale all the data to the same reference, let's use that). We split 80/20. I also set random_state (what we use to randomly pick which element goes into training or test set) to a set value, so you and I find the exact same results. In real life, you would not set the random_set.

```shell
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

The rest is "as usual". Now that we have a training set, let's call a model, and ask the best fit for our data. Note that in this case, we limit the iterations to 1000, simply because the set is so well known, that we know that we'll converge early, so no need to explore the best iteration count.

```shell
# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

Done. Please pause a second, look through the window, contemplate the immensity of the universe and how, with scikit learn, you just call the model in one line, then train it in a second line. Two lines, and you are done (again, you do need to load the data etc., but that's something you need to do anyway, with any tool).

So we are done, but is our model any good? An easy way to know is to ask the tool to make predictions for all elements in the test set. Then, we ask it to compare the prediction to the real classification, and output an accuracy score (which is a percentage). Then, we just ask it to print that score:


```shell
# Make predictions and evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
```

If your code is like mine, you should get a 100% accuracy score. Perfect score? Not too bad for a single line of code.

Some people find this simple score enough. Others like more visual representations. let's do both. Let's use pyplot to build the confusion matrix. If you do not know what a confusion matrix is, then you read too fast above, don't just look at the code, also read the text between the code. We also use seaborn, to build the nice heat scale on the side.

```shell
# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

This is great. But this model is now trained and in computer's Jupyter (volatile) memory under the name 'model'. But how would you export this model, in order to run it elsewhere (for example on ESP32)? Well, here where we have bad news. Microcontrollers can support a light version of Python (micropython), but the export of the model is a bit manual. You first need to extract the coefficients (or weights) and the intercept from the trained model (let's print them too so we can see them):

```shell
# Assuming 'model' is the trained Logistic Regression model
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)
```

Then, once you are in micropython, you'll need to manually implement the logistic regression prediction. This involves coding the sigmoid function and using the extracted coefficients and intercept to make predictions. Something like this:

```shell
# In MicroPython, 
# Here's a basic template of what the logistic regression function might look like in MicroPython:
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(features, coefficients, intercept):
    z = sum(features[i] * coefficients[i] for i in range(len(features))) + intercept
    return sigmoid(z)

#(replace coefficients and intercept in this function with the actual values you extracted from your Scikit-learn model)
```
You can type this in Jupyter, Python will understand it. Keep it also somewhere in case you need to implement this type of code in micropython one day.

Some micropython implementations support storing (and loading) a model in json format. In general, you can save the coefficients and intercept to a file or directly hardcode them into your MicroPython script. If you choose to save them, you can use Python's json module:


```shell
import json

model_params = {
    'coefficients': coefficients.tolist(),  # Convert numpy array to list
    'intercept': intercept.tolist()
}

with open('model_params.json', 'w') as file:
    json.dump(model_params, file)


# You can then transfer this model_params.json file to your MicroPython environment and load it there.
```


We will compare how you would do the same in tensorflow in a few paragraphs, so keep the output of your notebook to be able to comapre as we move down.

But before comparing with tensorflow, let's load a second example in scikit learn. A second example may look redundant, but you will see how it makes sense as we move along. This time, we use another classic, the [NMIST script dataset](https://www.tensorflow.org/datasets/catalog/mnist), which is a collection of 60,000 manually written digits, form 0 to 9. It is also a classic for image recognition. You will see in a later lesson that a standard type of AIML for such task is Convolutional Neural Network (CNN). Unfortunately, scikit learn does not support CNN directly, so we need to use another classifier. We can use SVM (go back to the logistic regression slides if the name does not ring a bell) to achieve kinda the same goal, but here we are really doing classification, not image recognition.

So let's load SVM, we will also import the time function, to see how long each computation takes:

```shell
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import time
```

Then we start the clock, load the dataset (re-orient it as it is not structured the way we need, this is what 'reshape' does), then split as usual. Next, we call the SVM classifier ('clf') and ask for the best fit. We then evaluate our fit on the test set, and output the model accuracy. All this is similar to what we just did above, with the difference that we use SVM. Then we stop the clock:

```shell
# Start timing
start_time = time.time()

# Load the dataset
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, predicted)

# End timing
end_time = time.time()

print(f"Scikit-learn SVM Accuracy: {accuracy:.2f}")
print(f"Execution Time: {end_time - start_time:.2f} seconds")
```

If your code is like mine, then you should have an accuracy of 0.97. The computation time depends on your machine, but keep track of it.

Now, before moving forward, save your Notebook (click the save icon at the top). This is important, as we are going to mess up things a bit. Once your notebook is saved, close the Jupyter tab.
Then, navigate back to Anaconda Navigator, and click the Environments menu on the left. You should get to a page like this. Yours may just have "base" in the list (and that's okay).


![alt test](beforetsf.png?raw=true "Anaconda Environments")

What we want to do here is install tensorflow. The bad way to do it is to just look for the package as we did in lab 1, and go install it in your base installation, if your base already has scikit learn. Both sets of libraries do not have the same dependencies, and by installing one (and its dependencies), you make it difficult to install another one. You will get dependency issues warnings, and if you go through an override, you won't be able to use the previous library anymore. So the right way to proceed is to install one new environement for tensorflow, and move between environments if you need to develop in one world or the other. In most cases (outside of this class), you will not develop a code that uses both scikit learn and tensorflow (or pytorch). If you do, there is something very wrong with your approach.

So we want to create a new environment. At the bottom of the list, click Create. Give a name to your environment (like Mytensorflow or any name that is meaningful to you in this context). The tool will ask you to install a Python version, 3.10 is a good pick (do not pick at random, as tensorflow has specific python requirements), then click create.

Once the environement is created, click it to activate it. In Anaconda, it should show a green Play icon (just like base had it in the image above). Once you are in that environment, on the right, look for the tensorflow package, and the tensorflow-datasets package. Select them and install them. They will require many dependencies, accept them and continue.

Once all the packages are installed, keep the tensorflow environment active, and go back to the Home tab.

There, select Jupyter Lab and launch it. 

Notice what just happened. You cannot run Jupyter notebook from one environment to the next (well, you 'can', but it is in general a bad idea). When you switch between environments, close Jupyter (save stuff before switching!) and reopen it from the next environment.

Go back to the Notebook you were using above. The results from the Scikit learn part should still be visible. However, do not run the computations again, as they would likely fail (for lack of the proper library).

Go to the bottom block, and continue from there.

We will start with the iris flower case, and run the same in tensorflow. If you forgot, go back up and retrace what you did for the iris case with scikit learn.

The first part is obvious and almost boring, if it wasn't so fun. We need to import the libraries we need, numpy, pyplot, seaborn. We will also need pandas, as we'll need the iris dataset from another source, and that source has it stored in a panda datframe format instead of a numpy array (like scikit learn did).

```shell
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
```

We also need to import a few tensorflow libraries. This is where things start to diverge from scikit learn. In tensorflow, we will need to build (declare) the model. This means that, instead of importing just the model type, we need to import the tools under the model structure (under the hood). This sounds intimidating the first time you do it, but as it is always the same, you just get used to it.

```shell
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
```

Next, we need the iris dataset. Tensorflow has plenty of models, but it pulls iris from its original source (instead of delivering a local copy like scikit learn did). No worries, and in fact if you look in the tensorflow catalog, they [tell you where they get it from](https://www.tensorflow.org/datasets/catalog/iris). So we load it, asn they suggest, we name the columns as they should be:

```shell
# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, names=columns)
```

Take a second to look at the panda dataframe structure:

```shell
iris_df
```

The next step is optional, but nice for visualization later on. Let's associate a name to each flower variant (they are just stored as numbers in the dataframe):

```shell
species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris_df['species'] = iris_df['species'].map(species_mapping)
```

Just like we did with scikit learn, we need to divide the dataset between the features (what we called X above) and the variant (the target, what we try to train and predict):

```shell
# Separate features and target
X = iris_df.drop('species', axis=1).values
y = iris_df['species'].values
```

One additional step we do here is to apply one-hot encoding to the categories. This is a way that helps with speed in tensorflow. One-hot-encoding also gives a classification (category 0, 1 or 2), but with a different presentation, where there is one column per category. So a flower of category 0 would be classified as [1., 0., 0.], one that belongs to category 2 would be [0., 0., 1.].  

```shell
# One-hot encode the target variable
y_encoded = to_categorical(y)
```

Just like for scikit learn, we need to put all the variables at the same scale. We cannot call the StandardScaler like we did in scikit learn (tensorflow does not have it), we have to do it by hand:

```shell
# Standardize the features
mean, std = np.mean(X, axis=0), np.std(X, axis=0)
X_scaled = (X - mean) / std
```
Next, we split the dataset into training/test. You can see that the idea is the same, but we need to shuffle the deck (while scikit learn did it implicitely):


```shell
# Split the dataset into training and test sets
np.random.seed(42)
shuffle_indices = np.random.permutation(len(X_scaled))
train_size = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[shuffle_indices[:train_size]], X_scaled[shuffle_indices[train_size:]]
y_train, y_test = y_encoded[shuffle_indices[:train_size]], y_encoded[shuffle_indices[train_size:]]
```

And here the main difference: we need to build the model. As there is no direct logistic regression function (we could write one by hand), we can use what tensorflow is very good at, neural networks. For now, this may sound strange, keep a note of what this says and come back to it once we have done the neural network lesson:

```shell
# Build the model for multi-class classification
model = Sequential([
    Dense(10, activation='relu', input_shape=(X.shape[1],)),
    Dense(3, activation='softmax')  # Output layer for 3 classes
])
```

Once we have defined the model, we need to compile it:

```shell
# Compile the model
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
```

And only now do we go back to what we were doing with scikit learn, train the model with a fit command. Note how we have to define batch sizes (how many flowers do we compare in one go), and epochs (how many rounds of comparison we do):

```shell
# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=0)
```

Once the model is trained, let's look at its accuracy:

```shell
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {accuracy:.2f}")
```

Wait a minute, what accuracy did you find? If your code is like me, 0.97. But with scikit learn we have 1.00. We are close, but on such a simple dataset, we should ace it. This is not good. Okay, we'll come back to that in a second.

Meanwhile, let's look at the confusion matrix. I am using a slightly different code from above, as confusion_matrix is also integrated into scikit learn, but not in tensorflow:

```shell
# predict the values:
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Initialize the confusion matrix with zeros
num_classes = len(np.unique(y_true))
confusion_matrix = np.zeros((num_classes, num_classes))

# Populate the confusion matrix
for i in range(len(y_true)):
    confusion_matrix[y_true[i], y_pred_classes[i]] += 1

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
```

Okay, you see where the code gets confused. 0.97 is just not good enough. Try to change the number of epochs, or the batch size, and see if you can get to 1.0

Yes, I hear you think: we use 'fit' so that we do not have to do it by hand, and here we are tuning by hand anyway. This is one limitation of neural networks, there is no good math to describe what is the right number of parameters, epochs, structure etc. for a given problem. However, playing a bit with the parameters also helps better understand what really happens. In this case, try to keep the batch size to 5, but set the epochs to 1000. The code should take longer to run, but you should get an accuracy of 1.0. This is a bit of a heavy approach, and something smaller than 1000 might work as well (and be faster).

Accuracy is central to AIML, because it determines how useful your model is. In general, you want to shoot for the highest possible accuracy. That is a problem we also saw above with the second case and SVM. If your code is like mine, you got 0.97. This is good in the real world. On this type of well-kown deck, it is a bit unfortunate that you can't get to 1.00, or at least 0.99. The question then is: can you get a different score (preferably better)? In the scikit learn approach, where everything is automated, getting higher is not obvious. If you read [scikit learn documentation on svm](https://scikit-learn.org/stable/modules/svm.html), you will find advice, a few ways to tune the engine, but no obvious parameter you can use to increase your score. This is where neural networks flexibility comes back to your mind, and also the fact than tensorflow requires you to define your model: it is not so much because the tensorflow people were too lazy to implement defaults in their libraries, but rather that they realised early that, as soon as the problem becomes a bit complex, you will need to tune your model a bit. So you might as well define it, so you can see the parameters at your disposal.

So let's go back to our second problem, and see how we can implement it with tensorflow. You should already have some libraries from before, but just in case, let's make sure you  have loaded tensorflow, but also a few libraries related to neural networks, like CNN (in 2D), pooling, flatten, and sense. These libraries may not mean anything to you yet, they will become clear after our class on neural networks:

```shell
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import time
```

Then, we will proceed like with SVM. We will first load the dataset, then split it into a training and a test sets. Here again, the format of the dataset is not exactly what we want, so we use reshape to change its layout (reshape allows you to change what data goes into a column vs. a row, define how many dimensions you want etc., it does not change the data, just the way data is displayed). 


```shell

# Load and preprocess the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```

Then, we need to declare the model. We input the images, which are 28x28 pixels in black and white (1 color). Our convolutional (CNN) network will then looks at blocks of 3x3 pixels to try to find patterns. We will use a relu layer, then will pool groups of 2x2 blocks. We will use a layer of 128 units, followed by another oen of 10 units. Again, take note of these elements if they do not make sense to you yet, to return back to them after our class on neural networks.

```shell
# Build the CNN model
model = tf.keras.models.Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

Once we have defined the model, we compile it, i.e., define our criteria for a succesful model. We use an optimizer called [Adam](https://keras.io/api/optimizers/adam/), which has the great property of keeping one learning rate per parameter (to take our linear regression example, that you may be familiar with, we then build one learning rate for tetha0, and another one for tetha1, making our optimization more accurate than one-size-fits-all learning rate). We also define how we compute the error, and what our target metric is (obviously, accuracy):

```shell
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

Then, all we have to do is to run the training, and evaluate its perfomances. We clock it to compare with the SVM approach:

```shell
# Start timing
start_time = time.time()

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=128)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

# End timing
end_time = time.time()

print(f"TensorFlow CNN Accuracy: {accuracy:.2f}")
print(f"Execution Time: {end_time - start_time:.2f} seconds")
```

If your code is like mine, you should get, just with this first attempt and 5 epochs, an accuracy of 0.99. And you know that tuning the ecpoch count (try it) will increase the accuracy. Depending on your machine, the computation time may be more, or may be less, than with SVM. The main difference lies in your machine. Neural networks take great advantage of GPUs, and can run lightning fast on these architectures. 

# Converting to tflite

Just like their scikit learn counterparts, tensorflow models can be saved. A major difference is that tensorflow also exists in the tflite version, which can run on microcontrollers. This is what the Espressif demo models run, and it is useful to know how to convert to a tflite model. This is done in a few simple steps. First you need to save your model, so you can reuse it. Then, you take that saved model and convert it to tflite format. Last, you save the tflite model.

```shell
# Assuming 'model' is the trained TensorFlow model

# Step 1: Save the TensorFlow model
model.save('my_model')

# Step 2: Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model('my_model')  # Path to the saved model
tflite_model = converter.convert()

# Step 3: Save the TFLite model
with open('my_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

Let's look at that model. There are 3 common ways to do it. The first one is via a site: go to (netron)[https://netron.app/] load your tflite model, and have a look at the structure analysis. Give it a try. This is great if you do not have your tools readily available, a website does the job for you.

Another way is TensorFlow Lite Interpreter Python API, a command-line tool that's part of the TensorFlow Lite library. It provides detailed information about the model, such as its size, input/output tensors, and operations. To use this tool, you need to have TensorFlow installed. You can analyze your model using the following command:

```shell
import tensorflow as tf

# Load the TFLite model
tflite_model_path = 'my_model.tflite'  # Replace with your model's filename
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input and output details
print("Input Details:", input_details)
print("Output Details:", output_details)
```

A third way, if your TFLite model includes metadata, is to use the TensorFlow Lite Metadata library to read this metadata programmatically. The metadata can include information about model inputs/outputs, labels, and associated files. In our simple case, the command won't output much, as expected (because it does not include metadata).

```shell
# You need tflite support, if you get an error message, install it with !pip install tflite-support


from tflite_support import metadata as _metadata

# Load the model and read metadata
displayer = _metadata.MetadataDisplayer.with_model_file('my_model.tflite') # replace with your model name
print(displayer.get_metadata_json())
```



