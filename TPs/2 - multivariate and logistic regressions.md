# Multivariate regression

In the previous exercise, we explored linear regression in the case of of an oil pump. Let's explore this data a bit further. Open a new Jupyter notebook, and start by loading the basic libraries:

```shell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
```

Also load the csv data we used last time. Again, change the below to the path on your machine:

```shell
df = pd.read_csv('/Users/jerhenry/Documents/Perso/IMT/IMT_ML_IoT/unconv_MV_v5.csv');
```

When loading a new dataframe always start by looking at your data, to check that everything looks like what you expected:

```shell
df.head()
```
In the previous exercise, we observed that there was a linear relationship between Por and Prod. Por is indeed related to Prod, but it is also related to another parameter. Spend some time graphing Por as x against the other columns (as y), you will find one of them that also 
displays a linear relationship with Por. Besides Por (molecular porosity) and Prod (production output) that you already know, the set includes Perm (permeability, how well water can mix with the oil), AI (accoustic impedance, how well sound traverses the product), 
Brittle (brittleness of hard particles), TOC (total organic carbon), and VR (reflectance).

Which other linear relationship did you find to Por? Graph it and find the coefficient and the intercept. Keep each individual relationship in your Jupyter Notebook (and share your notebook with me). Use the previous exercise techniques to graph the relationships.

We also want to explore multivariate relationships, in particular between Por, Prod (which we already studied), but also TOC (total organic carbon).

```shell
plt.figure(figsize=(18,12))
# A fist step is to look at if TOC is also linear with Prod
plt.plot(df[['TOC']], df[['Prod']], 'o')
```

You should also see an appearance of linear relationship. It would be interesting to find the slope of the linear relationship. We are in 3D, so the slope will be along (x,y,z), but the principles are the same as in previous exercise. 
You could decide to run the linear regression manually as we did before, but the more I think of this possibility, the more I find myself staring at the wall, thinking that I am giving you the choice between a car and a pile of car parts (to build your car yourself), and wondering
in what universe you would choose the pile of parts... so I am going to assume that you will want to use the tools, and run scikit learn libraries. Let's load them:

```shell
from sklearn import linear_model
# let's create the linear regression object
reg = linear_model.LinearRegression()
```

Then, you 'just' need to ask for the best fit:

```shell
# The fit is linear, we can thus set our model with 2 variables
reg.fit(df[['Por', 'TOC']],df[['Prod']])

```

Well, that was fast. Let's look at the coefficients and intercept.

```shell
int(reg.intercept_), reg.coef_

```

Just like in the previous exercise, it is nice to be able to plot the line. This is in 3D, but the principles are the same as in the rough 2D graphing code. We create 2 points at the edge of the x domain, then predict their y and z value, and draw a line going through these 2 points.

```shell
fig1 = plt.figure(figsize=(18,12))
ax = plt.axes(projection ='3d')
ax.scatter(df[['Por']], df[['TOC']], df[['Prod']])
ax.set_title('Optimal pressure based on Por and TOC')
ax.set_xlabel("Viscosity (Por)")
ax.set_ylabel("Granularity (TOC)")
ax.set_zlabel("Pressure (Prod)")
A5 = 5.5
B5 = -0.2
C5 = int(reg.predict([[A5, B5]]))
A6 = 24.5 
B6 = 2.2
C6 = int(reg.predict([[A6, B6]]))
P5 = [A5, A6]
P6 = [B5, B6]
P7 = [C5, C6]
ax.plot(P5, P6, P7)
plt.show()

```

Reproduce the same process (fit, intercept and coefficients display, graph) with the other values that you found correlated to Por.

A good idea is also to save this model, so you do not have to re-run the training each time (and on all machines). There are multiple ways of doing it, and you will learn more ways in the upcoming exercises. let's start simple, with a library that does exactly that: save and load models. It is called joblib. You will find a lot of debates online about which libraries does that job best, and if joblib is more secure than XYZ. For now, let's ignore the debate, and just save our model as we need:

```shell
import joblib
# Save your model to a file - you should see that file in your working directory
joblib.dump(reg, 'my_cool_joblib_model')

```

Now did it work? The best way to know it to try to load and use your saved model. Before doing that, and while your model is still in Jupyter, predict some values. For example:

```shell
reg.predict(([[5.5, -0.2]]))

```

Now that you know what the prediction is, make Jupyter forget everything. An easy way to do this is to go to the top of the Jupyter window and click the circular arrow (reload the kernel). Alternatively, in the top menu, go to Kernel and select Restart the kernel.

Now if you ask for a prediciton, Jupyter should not even know what reg is about:

```shell
reg.predict(([[5.5, -0.2]]))

```

Now you may need to reload support for the libraries we use in this part of the exercise:

```shell
from sklearn import linear_model
import joblib

```

Now, you can reload your model, and the prediction should work again (not only it should work, but it should give you the same result as above):

```shell
mj = joblib.load('my_cool_joblib_model')
mj.predict(([[5.5, -0.2]]))

```

# Logistic regression in Jupyter

Logistic regression does not seek to find the equation that best describes your data, but find which data belongs to which group. The csv file contains data useful to our purpose here, namely the record of when the pump ended up being clogged after the pressure was changed. As usual, it is great to visualise data. You may want to go to the first block of your notebook and reload the standard libraries (numpy, pyplot, etc.) along with the csv file, as we will need them here as well. Then, let's plot the columns of interest:

```shell
plt.figure(figsize=(18,12))
plt.plot(df[['Brittle']], df[['Reuse']], 'o')
plt.title("Pumps issues based on grains brittleness")
plt.xlabel("Brittleness")
plt.ylabel("1 if pump could be reused, 0 if it was clogged")

```

As you can expect, scikit learn incorporates the libraries for logistic regression. We didn't do this for the previous part of the exercise, but as we go further, we want to incorporate more and more good practices. One of them is to split the dataset into a training set and a test set. You can do it manually, but you would also expect that there is a simple command for that. And there are many, one of them in scikit learn. You use it by defining what is the training percentage (below, 80% for training, 20% for testing), then calling out the names of the training and test parts (for the X and Y values, when you run a single variable case as we do here):

```shell
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df[['Brittle']], df.Reuse, train_size=0.8)

```

Next, you can load the logistic regression library. Above, we called the model 'reg', then 'mj'. The name does not matter, as long as you remember it. Also, in a notebook where you use different models, it is also a good practice to give them different names, so that you know which one you are calling. So let's name this one 'model':

```shell
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

```

Now, just as before, all we have to do is to call the 'fit' command on our data. As we split the data between a training and test sets, we call the fit command on the training part (as the goal, with fit, is to train the model):

```shell
model.fit(X_train.values, Y_train)

```

Now if we used fit, we found coefficients. Let's have a look at them:

```shell
model.intercept_, model.coef_

```

For a logistic regression task, the coefficients may not mean much to you. An easier way is to look at the model performances on the test set. First, have a look at the real data (did the pump get clogged [1] or not [0]):

```shell
Y_test

```

Now compare with the model predictions (for the X_test values). If the model worked, most of the predicted y values should be the same as the real y values you just saw:

```shell
list(model.predict(X_test.values))

```

The output is simplified (0 or 1). In the real world, you may want to see the real prediction, which is a probability value (probability of 0, probability of 1). In this simple case, the values are very close to the simplified version, but at least you can see the real probabilities:

```shell
model.predict_proba(X_test.values)

```

To illustrate how the prediction works by projecting the probability onto a curve, we can generate brittleness values (from 10 to 85, by jumps of 0.5), then plot the prediction:

```shell
brittleness = np.arange(10, 85, 0.5)
probabilities= []
for i in brittleness:
    p_clogs = model.predict_proba([[i]])
    probabilities.append(p_clogs[:,1])
plt.scatter(brittleness,probabilities)
plt.title("Logistic Regression Model")
plt.xlabel('Brittleness')
plt.ylabel('Status (0: clogged, 1: reused)')

```

# Logistic Regression in ESP32

Logistic regression is a classification effort. As long as you have measurements (a series of numbers) that you can compare to two sets (each with a series of numbers), you can decide which set you measurements are closest to. One basic example is motion detection. Another one is basic speech recognition. Let's explore them in turn, motion here, and speech in next lab.

## Motion Detection from ESP-WHO

This lab supposes that you have installed edp-idf v4.4.x (NOT 5.x), preferrably 4.4.2. 

If you do not have ESP 4.4.x installed yet, create a working directory (e.g., ~/IMT/esp-idf), then create an esp sub-directory and clone esp-idf v4.4.2 there:

```shell
$ mkdir -p ~/esp
$ cd ~/esp

$ git clone -b v4.4.2 --recursive https://github.com/espressif/esp-idf.git esp-idf-v4.4.2
```

Once the cloning completes, run the esp installation script. You really only need the esp32 support, but if you want support for more platforms, replace 'esp32' below with 'all':

```shell
$ cd esp-idf-v4.4.2/
$ ./install.sh esp32
```

As you will write over the USB interface, you need to make sure that you have the right to do so. If you connect your ESP-EYE to the machine USB port, you should be able to check which port the board connects to. Not plugged yet:

```shell
$ ls /dev/ttyUSB*
$ 
```
Then, ESP-EYE plugged into the USB port:

```shell
$ ls /dev/ttyUSB*
/dev/ttyUSB0
```

If you are under Linux (skip this if you are under MacOS), add yourself to the dialout group to make sure that you have the right to write to that USB port:

```shell
$ lsudo usermod -a -G dialout $USER
```

At this point, you have ESP-IDF. You then need to install ESP-WHO, which is a small set of examples and libraries for optimized AIML (primarily neural networks, but this detail does not matter yet) on ESP32. You need the ESP-WHO version that matches the esp-idf v4.4.x you installed. Go back to your root directory (e.g. ~/IMT/esp-idf) and clone ESP-WHO git there:

```shell
$ git clone -b idfv4.4 --recursive https://github.com/espressif/esp-who.git
```

Once the install completes, navigate to the esp-who example folder:

```shell
$ cd esp-who/examples 
```

Use the 'ls' command to check the content. You will see different subdirectories. Each example includes different subfolders, with the calls required to compile the example for the terminal, LCD or web interfaces. Not all platforms support all interfaces. Navigate to the motion detection web interface folder:

```shell
$ cd motion_detection/web
```

As the work is pre-done, all you have to do is build the executable code, and push it to the ESP-EYE. Start by setting the target as ESP32:

```shell
$ idf.py set-target esp32
```

This command builds the execs for your platform. All you have left to do is push the code to the ESP-EYE, and check on the CLI that the code is running:

```shell
$ idf.py flash monitor
```

After a few seconds, you should see on your terminal that the app has started:

```shell
I (1016) camera_httpd: Starting web server on port: '80'
I (1016) camera_httpd: Starting stream server on port: '81'
I (1026) main_task: Returned from app_main()

```

From your laptop, look for the available list of Wi-FI SSIDs. You should see one named motion detection. Connect to that SSID. The SSID runs on the ESP32, and will provide to your laptop an IP address (192.168.4.2). Open a web browser to your ESP32 web server (at 192.168.4.1). You should see the main interface. Start the stream, and move the camera (or something in front of the camera), you should see that motion is detected.

## What happens in the background?

In the motion_detection/web/main folder, you will see an app_main.cpp file, that calls who_motion_detection.hpp. 

Then, when the camera is on, it continuously takes an image (at a pace you can configure). Whatever the resolution, the image is converted to grayscale (320x240 pixels), then split up in to 20x20 pixel blocks (so you end up with 192 blocks for each image). The pixels have a grayscale color property, which you can think of as brightness value. The program then takes the color of each pixel in each block, and merely computes a mean value. Thus each block ends up with an average 'color' or 'brightness' value, which is a number between 0 and 255. Then the grayscale number for each block in an image is compared to the grayscale number of that same block in the previous image. If the numbers are different by more than a certain value (by default '7', and you see how by changing this number you can set the motion detection sensitivity), then this block is declared changed. If enough blocks have changed (by default 15 or more) then motion is declared.

Is this machine learning? No. It could be, if there was a model trained on images compared two by two, with a classification ('there was inded movement' or 'not'), the model would then have learned these numbers (7 and 15) in a more dynamic way. It would have been more difficult, because the model would have found that some blocks are less important than some others. We will see how such training is done using convolutional neural networks (CNN) in a later lesson. The main point here is to remember your goals: you do not necessarily need an AIML engine to achieve that goal: a simple script may be sufficient. AIML may bring higher reliability, but at the cost of higher complexity.


