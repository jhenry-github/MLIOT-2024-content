# Rock Paper Scissor Game under ESP-EYE - AN with image recognition exploration

This exercise aims at implementing a rock, paper scissor game on ESP-EYE (the general structure is borrowed from (hackster.io)[https://www.hackster.io/]). You will see more in details how the implementation works as we explore its structure below, but in short, we use one of [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors), 
with a small UI wrapper (we use ESP-EYE terminal to keep the code small), and tensorflow lite. Training the model is done on your computer, then the trained model and tflite are pushed to ESP-EYE. The inference is then run on ESP-EYE, where you show your hand (paper, rock or scissor), 
the neural network on ESP-EYE analyzes the image, finds the most probable match (rock, paper or scissor), then shows you that it found the right match by selecting a random gesture that may (or not) beat yours (so, if you are new to the game, paper beats rock, rock beats scissors, and scissors
beat paper).

## Installation

This lab is best achieved on Linux, where you will have the least amount of compatibility issues between Python, the esp-idf version you will use etc., or on MacOS.

If you are on Linux, start by making sure that you have Python, at least 3.10 (if you are on Mac, jump to the first itemis git clone command below). I have tested sucesfully with Python 3.10, 3.11 and 3.12. You can check your Python version with:

```shell
$ python --version
Python 3.11.7
```

If you see something older, you can install the latest Python version,, or a specific version. To install a specific version, you need to make sure you have a few repositoris:

```shell
$ sudo apt install software-properties-common
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt update
```

Then, you can install the Python version of your choice, for example 3.11:

```shell
$ sudo apt install python3.11 -y
```

You can then verify that the installation worked:

```shell
$ python3.11 -V
Python 3.11.7
```

If you want to run several Python versions in parallel, you can use updates-alternatives, for example, supposing you have 3.10.8 and 3.11.7:

```shell
$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2
```

The latest entry is what will be called when you type 'python' on the command line. You can switch between them with:

```shell
$ sudo update-alternatives --config python
```

You will get a menu looking like this:

```shell
 0            /usr/bin/python3.10   0         auto mode
* 1            /usr/bin/python3.11   0         manual mode

Press  to keep the current choice[*], or type selection number:1
```

Jump here if you are on MacOS. Once you have Python installed, copy the project source provided by Itemis. Position your terminal in a folder, for example ~/IMT/esp-idf, then clone the project:

```shell
$ git clone https://github.com/itemis/tflite-esp-rock-paper-scissors
Python 3.11.7
```

Spend a few minutes looking at the project files. You will see that most of the key files are under src/

```shell
$ ls
ARCHITECTURE.md  CONTRIBUTING.md  img  LICENSE.md  LOAD_COLAB.ipynb  poetry.lock  pyproject.toml  README.md  requirements.txt  src
$ ls src
data_collection  data_preprocessing  keras_model  model_pipeline.sh  tf_lite_model  tinyml_deployment
```

The details may look a bit vague for now, but they will become clearer soon. 

In order to be compilable and trainable, the project needs a few Python standard tools, like numpy, but also of course tensorflow and keras. However, the default list of requirements is faulty, calling for sklearn (which is the old name for scikit-learn). So one good step is to look at the list of requirements, 
but also fix it. Use vi or nano to edit the requirements.txt file (I use vi below):

```shell
$ cd tflite-esp-rock-paper-scissors
$ vi requirements.txt
```

In the optional section, you will see sklearn, replace it with scikit-learn, to look like this:

```shell
# optional
opencv-python
keras_tuner
scikit-learn
matplotlib
jupyter
selenium
pydot # requires graphviz https://www.graphviz.org/download/
```

Note that the project mentions "essential", "optional" and "development" tools. It is true that you could run the project without the 'optional' tools, but you would have to edit the project files. This class is about AIML, not Python development, so for our purposes all the tools are needed.

```shell
$ python -m pip install -r requirements.txt
```

We will soon get to the esp-idf part. Meanwhile, what we know we will need for the ESP-EYE task is support for the esp32 camera, tflite, and support for a small neural engine compute. These elements are listed in a file called update_components.sh, that you need to call, to make sure that the
latest version of these tools is in your project repository, ready to be compiled with the rest. Make sure that the file is executable, and call it:

```shell
$ chmod +x src/tinyml_deployment/update_components.sh
$ ./src/tinyml_deployment/update_components.sh
```

The update should go without any specific issue, apart from the possible request for adding dependencies. Another dependecy you will need will be esp-idf. You may have esp-idf installed already on other systems, but you need to be aware that there is a major difference betwen the v5.x and v4.x. For this program to work, you need to make sure you install and run 4.4 (not 5.x), In fact, it is best if you run the one we tested with, 4.4.2. First, make sure you have all the dependencies needed for esp-idf:

```shell
$ sudo apt install git wget flex bison gperf python3 python3-venv cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
```

Then, still from the default directory you created earlier (e.g., ~/IMT/esp-idf), create an esp sub-directory and clone esp-idf v4.4.2. there:

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

As you will write over the USB interface, you need to make sure that you have the right to do so. If you conenct your ESP-EYE to the machine USB port, you should be able to check which port the board connects to. Not plugged yet:

```shell
$ ls /dev/ttyUSB*
$ 
```
Then, ESP-EYE plugged into the USB port:

```shell
$ ls /dev/ttyUSB*
/dev/ttyUSB0
```

If you are in Linux, add yourself to the dialout group to make sure that you have the right to write to that USB port:

```shell
$ lsudo usermod -a -G dialout $USER
```

At this point, you have all you need: the esp32 idf, that will allow you to compile a code for the esp32 camera, the tflite add on, the rock/paper/scissor pre-trained model, and basic CLI wrappers for the user interface. You can build the code. 
First go into the tinyml_deployment folder where all the source code is located, ready to be built:

```shell
$ cd src/tinyml_deployment/
```

From here, you need to call the path of the esp-idf utilities, to be able to call them directly. Supposing that you installed esp-idf in ~/IMT/esp-idf/esp, from the tinyml_deployment folder, run:

```shell
$ alias get_idf='. $HOME/esp-idf/esp/esp-idf-v4.4.2/export.sh'
```
If you want this shortcut to be saved to your bash profile, so you don't have to create if for every project, save it to the .bashrc/.zhrc file:

```shell
$ cat >> .bashrc << EOL
> alias get_idf='. $HOME/esp-idf/esp/esp-idf-v4.4.2/export.sh'
> EOL
```

You need to close and re-open the terminal window, if you use the .bashrc direction, as the field is read when you login and open a terminal. Then, still from the tinyml_deployment folder, build the executables for the esp32:

```shell
$ idf.py build
```

If everything above went well, the command should complete succesfully and you have a build ready to go. If things went wrong and you get exception messages, well, now is the time to troubleshoot, as "it should work". If you get desperate, the content of the tinyml_deployment folder is [here](https://www.dropbox.com/scl/fi/5k3foauv6wulw7k5dghnk/tinyml_deployment.zip?rlkey=mdiwvicmdqtfguel4g62tyfuf&dl=0). Unzip it, it has all the files that should be there after successful build. Use this as a last resort measure, as the primary goal is to succeed on your machine.

## Running the model

Once the files are built, flash the ESP-EYE, and start monitoring its console. The flashing tool should detect the USB port where ESP-EYE is connected, but you can also state the port specifically, with the -p option:

```shell
$ idf.py -p /dev/ttyUSB0 flash monitor
```

Once the process completes, the game auto-starts, then loops around until you are ready to play:

```shell
I (996) spi_flash: flash io: qio
I (1002) cpu_start: Starting scheduler on PRO CPU.
I (0) cpu_start: Starting scheduler on APP CPU.
I (1011) spiram: Reserving pool of 32K of internal memory for DMA/internal allocations
The game is about to start!

################## first round! #################
3!
2!
1!
Show your hand!
I (6021) gpio: GPIO[13]| InputEn: 1| OutputEn: 0| OpenDrain: 0| Pullup: 1| Pulldown: 0| Intr:0 
I (6021) gpio: GPIO[14]| InputEn: 1| OutputEn: 0| OpenDrain: 0| Pullup: 1| Pulldown: 0| Intr:0 
I (6031) gpio: GPIO[5]| InputEn: 1| OutputEn: 0| OpenDrain: 0| Pullup: 1| Pulldown: 0| Intr:2 
I (6041) cam_hal: cam init ok
I (6041) sccb: pin_sda 18 pin_scl 23
I (6061) camera: Detected camera at address=0x30
I (6061) camera: Detected OV2640 camera
I (6061) camera: Camera PID=0x26 VER=0x42 MIDL=0x7f MIDH=0xa2
I (6151) esp32 ll_cam: node_size: 3072, nodes_per_line: 1, lines_per_node: 4, dma_half_buffer_min:  3072, dma_half_buffer: 12288, lines_per_half_buffer: 16, dma_buffer_size: 24576, image_size: 18432
I (6161) cam_hal: buffer_size: 24576, half_buffer_size: 12288, node_buffer_size: 3072, node_cnt: 8, total_cnt: 1
I (6171) cam_hal: Allocating 9216 Byte frame buffer in PSRAM
I (6171) cam_hal: cam config ok
I (6181) ov2640: Set PLL: clk_2x: 0, clk_div: 3, pclk_auto: 1, pclk_div: 8
Camera Initialized

interpretPrediction: paper=0 rock=0 scissors=0.996094
interpretPrediction player_move: 2
interpretPrediction: paper=0 rock=0 scissors=0.996094
interpretPrediction player_move: 2
interpretPrediction: paper=0 rock=0 scissors=0.996094
interpretPrediction player_move: 2
interpretPrediction: paper=0 rock=0 scissors=0.996094
interpretPrediction player_move: 2
interpretPrediction: paper=0 rock=0 scissors=0.996094
interpretPrediction player_move: 2
classcount:
0 0 5 
---- ---- ---- ----
AI plays: paper!

 .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |    _______   | || |     ______   | || |     _____    | || |    _______   | || |    _______   | || |     ____     | || |  _______     | || |    _______   | |
| |   /  ___  |  | || |   .' ___  |  | || |    |_   _|   | || |   /  ___  |  | || |   /  ___  |  | || |   .'    `.   | || | |_   __ \    | || |   /  ___  |  | |
| |  |  (__ \_|  | || |  / .'   \_|  | || |      | |     | || |  |  (__ \_|  | || |  |  (__ \_|  | || |  /  .--.  \  | || |   | |__) |   | || |  |  (__ \_|  | |
| |   '.___`-.   | || |  | |         | || |      | |     | || |   '.___`-.   | || |   '.___`-.   | || |  | |    | |  | || |   |  __ /    | || |   '.___`-.   | |
| |  |`\____) |  | || |  \ `.___.'\  | || |     _| |_    | || |  |`\____) |  | || |  |`\____) |  | || |  \  `--'  /  | || |  _| |  \ \_  | || |  |`\____) |  | |
| |  |_______.'  | || |   `._____.'  | || |    |_____|   | || |  |_______.'  | || |  |_______.'  | || |   `.____.'   | || | |____| |___| | || |  |_______.'  | |
| |              | || |              | || |              | || |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 

You play: scissors!
The winner is you.
---- ---- ---- ----
Cover the camera to start the next round.
Waiting ...
################## next round! #################
3!
2!
1!
Show your hand!

```

To play, position your hand in front of the of the camera during the count down, and make a paper, rock or scissor gesture. The ESP-EYE will take five pictures after the "show your hand" output, then run the tflite small neural network to detect which gesture is the most likely for each picture. The system will then run a mini-random forest to pick the most lilely vote. Meanwhile, the ESP-EYE also randomly picked one of the gestures, then tells you if you won or the ESP-EYE won.

```shell
Show your hand!
interpretPrediction: paper=0 rock=0 scissors=0.996094
interpretPrediction player_move: 2
interpretPrediction: paper=0.0195312 rock=0.980469 scissors=0
interpretPrediction player_move: 1
interpretPrediction: paper=0.285156 rock=0.714844 scissors=0
interpretPrediction player_move: 1
interpretPrediction: paper=0.00390625 rock=0.996094 scissors=0
interpretPrediction player_move: 1
interpretPrediction: paper=0 rock=0.996094 scissors=0
interpretPrediction player_move: 1
classcount:
0 4 1 
---- ---- ---- ----
AI plays: scissors!

 .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. |
| |  _______     | || |     ____     | || |     ______   | || |  ___  ____   | |
| | |_   __ \    | || |   .'    `.   | || |   .' ___  |  | || | |_  ||_  _|  | |
| |   | |__) |   | || |  /  .--.  \  | || |  / .'   \_|  | || |   | |_/ /    | |
| |   |  __ /    | || |  | |    | |  | || |  | |         | || |   |  __'.    | |
| |  _| |  \ \_  | || |  \  `--'  /  | || |  \ `.___.'\  | || |  _| |  \ \_  | |
| | |____| |___| | || |   `.____.'   | || |   `._____.'  | || | |____||____| | |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------' 

You play: rock!
The winner is you.
---- ---- ---- ----
Cover the camera to start the next round.
Waiting ...

```

Please feel free to play the game forever. If you ever get tired of it, you can interupt the process with Crtl + ].

## Rebuilding the model, with your own data

Rock, paper, scissor is a fun game, but in the context of this class, what matters is the fact that a neural network runs on your ESP-EYE. One goal is to be able to reproduce the structure, with your own data. There are online tools to help you (like edge impulse, explored in another exercise).
However, it is highly desirable to have the ability to be able to construct the tool by yourself, so you can feed your own data, and understand how the engine works. You can work from any data a sensor, conencted to ESP32, can produce. As ESP-EYE has video and audio input, we continue with the video approach, 
to stay in the same line as the game above. For example, you way want to train your ESP-EYE to count your fingers, and perform some action based on the number of fingers you show: 1 finger -> turn on the light; 2 fingers -> ring an alarm etc. The second (and easy) part of this type of project is a small script, that, based on the predicted number of fingers, calls a command to perform the tasks you want. This part is not AIML and we'll skip it. 

The first part is to come to the conclusion (how many fingers?). This is the part we are interested about. 

### Collecting data

The first element of the model is of course data to train on, which means many pictures of your hands showing 1, or 2, or 3... 5 fingers. A general rule of AIML is that a larger set will result in better prediction. You should have at least 100 images of each finger position, but it won't hurt if you have thousands (apart that it will take you more time to take the pictures, and more time for the model to train). A common misunderstanding is that more training images will make the model larger. This is only partially true, and not true for the reasons people usually think. The model does not end up storing each individual image. In the training phase, the engine will chop the images in small sections (e.g., 3x3 pixels) and learn what feature needs to be there for each gesture to be correctly identified (for example, irrelevant background for the 1-finger gesture, part of your finger for the 2-finger gesture, etc.) Training on more images provides the model with better data on each zone. It may make the model a bit larger, because the training may help the model identify elements that are relevant, but that were missed with less images, but it is an acceptable price to pay for better accuracy.

So use your laptop camera (or your phone) to take pictures of your hand with the various finger positions. In an ideal world, use different lighting, different backgrounds, different shirt sleeves etc. so that the only common elements will be your hand with the right number of fingers visible. Save the images to png format. 

Alternatively, you may want to use the camera to shoot a video of your hand, and then split the video into individual still images. This approach saves you time, but be careful: a few seconds of video can get you an easy 100 images. However, they will likely show a single position, on a single background, with a single lighting condition. Resist the temptation to stop there, and invest more time to get different backgrounds, etc. as described above. Otherwise, you may train a model that will not recognize your hand as soon as the wall has a different color. To convert the video into still images, install ffmpeg, and run:

```shell
ffmpeg -i OneFinger.mp4 -vf fps=30 images%d.png
```

Run this conversion for each hand position (make sure not to mix the different positions in the same folder). Make sure that each hand position has about the same number of images. If one position has many more images than the others, you will be training a biased model, that will learn that most images are postion XYZ anyway, that it will recognize well, but will be much weaker on the other positions. Difference of a few images is fine.

Then, create a project folder. Inside the project folder, create a data folder, with one sub-folder for each hand position. Be mindful that the name of the folder will be what the engine will surface. So, for example, if one folder is called OneFinger, and contains all the images when you raise a single finger, when the engine will conclude that the image you are trying to analyse is likely the single finger image, it will surface OneFinger, the name of the folder where the images with the highest probability reside. So make sure to give a clear name to your folders!

```shell
$ mkdir ~/esp-idf/tflite-finger-count
$ cd ~/esp-idf/tflite-finger-count
$ mkdir -p data/raw_images/OneFinger
$ mkdir -p data/raw_images/TwoFingers
$ mkdir -p data/raw_images/ThreeFingers
$ mkdir -p data/raw_images/FourFingers
$ mkdir -p data/raw_images/FiveFingers
```

Then save your individual .png images to their respective folder (all images of your hand raising one finger into the OneFinger folder, etc.)

In order to be processed, the images need to have the same resolution. Clearly, 4K or 8K images are great, but would cause too many features to be stored by your model (imagine a feature as, for example, a 3x3 pixel zone of your image that is typical of what the image shows). In the case of the finger count, the subtle lines of your palm are of low importance, so you don't need an image with many features. If you have too many features in an image, running the inference (and the training) takes longer, because there are more feautures to compare. In fact, for ESP-EYE, a good image size is 96x96 pixels in grayscale. Anything bigger can work, but will work slower until the processing time becomes too high for the tool to be usable.

So your next step is to convert your images to 96x96 in grayscale.

Once the conversion is done, you want to train your model using the same neural network as for the rock, paper, scissors example. As you train your model you also want to evaluate the model efficiency. In order to do so, you may want to split your data set (your images) into a larger percentage used for training, and anotehr used for testing (for example 80% / 20% ratio). 

All these functions can be performed with standard python commands that you may remember from the lecture. If your memory fails you, the great news is that the rock, paper, scissor project includes the python scripts that were used for the project. Most of them are in the src/data_preprocessing folder. There, you will find python scripts like balance_classes.py, that will count the images in each folder, and make sure that they are the same (if one folder has more images, the script will randomly delete images from that folder until the counts between folders match). You will also find the preprocess.py script, that converts all your images into 96x96 grayscale format. You will also find the split.data.py script, that creates the 80/20 split for training and testing. You can call these scripts individually, or you can use the wrapper preprocessing_pipeline.py, that will call these other scripts in sequence. To run this wrapper script (or the individual scripts), go to the project folder. Thre should be there a data/raw_images folder, with subfolders matching your categories, each subfolders with .png images.Then call the script, for example (supposing the tflite-finger-count project is in the same folder as tflite-ep-rock-paper-scissor):

```shell
$ python ../tflite-esp-rock-paper-scissors/src/data_preprocessing/preprocessing_pipeline.py 
```

The script may require you to install additional dependencies, if you didn't do it earlier, like opencv-python.

Note that the data_preprocessing folder also includes and augment.py script. This script is useful if you do not have enough images in your data set. It will take each individual image and generate new images from it with random noise in it. (This paper)[https://paperswithcode.com/method/imagemorph], for example, uses this technique and explains it. This is useful if you cannot get enough images of a particular type, in real life. In this exercise, unless you can't take pictures of your hand anymore for some reason, it is better to get the real source, and get more pictures of your hand. But keep the augment.py script in mind, it may be usueful to you some day.

Once your pre-processing is ready, the next step is to train the model. Here again, there is a script to rule them all, that calls the other script, jump to it below if you are in a hurry, but if you can, spend time going step by step. The first step is to train the model. The training uses a convolutional neural network called (resnet50)[https://arxiv.org/abs/1512.03385], for Residual Network, with 50 leayers. In details, the rock, paper, scissor src/keras_model contains the python scripts you need to run the training. In effect, the training is a mix of full training, and training on the output layers of a model from your images, with the goal of speeding up the process (and the expectation that the feature recognition is very basic, as there are easy-to-spot differences between 'one' and 'two' finger images). Spend some time looking at the scripts in src/keras_model if you want to understand the details a bit further. Once you are ready, still from the tflite-finger-count folder, call the main.py script in the keras_model folder:

```shell
$ python ../tflite-esp-rock-paper-scissors/src/keras_model/main.py 
```

The script should run, and save your model under the bin_model/keras_model folder. Your next step is to convert that model (that is now tensorflow) into the smaller tflite format. Here again, this is a simple python command, but the tf_lite_model folder already has the script:

```shell
$ python ../tflite-esp-rock-paper-scissors/src/tf_lite_model/convert_to_tflite.py
 
```

ESP-EYE does not use the tflite model directly (with a linear structure representation), but instead uses a C data array (this does not change the model itself, just the way it is represented). Conversion is as easy as:

```shell
$ chmod +x ../tflite-esp-rock-paper-scissors/src/tf_lite_model/tflite_to_c_array.sh
../tflite-esp-rock-paper-scissors/src/tf_lite_model/tflite_to_c_array.sh
 
```

This will generate the bin_model/model_hexgraph.cc file. At this point, you are one conversion away from the build data you need. Obviously, deploying the model to the ESP-EYE also means sending there all the elements the board needs to run, from the bootloader to the camera drivers etc. One option is to copy all these elements from the rock, paper, scissor directory (and sub-directories) to your project. Another one is to copy that model_exgraph.cc file to the rock, paper, scissor directory (namely, under the keras_model folder), then compile from there. Both options are equal. Once the file has found its place, you need to convert the C array into a C/C++ code file, by running the model_to_mcu script:

```shell
$ python3 src/tf_lite_model/model_to_mcu.py

```

At this point, you are ready to compile and deploy your code, with your data, to the ESP-EYE. The code is the same as in the rock, paper, scissor case:

```shell
$ cd src/tinyml_deployment && get_idf && idf.py build && idf.py -p /dev/ttyUSB0 flash monitor

```

In this simplified exercise, the wrapper is still the same, the system will ask you to show your hand. Instead of interpreting it as rock, paper, scissor, it will display the OneFinger and alike label (but will still try to play its own hand, as the action taken from the recognition part, as explained above, is of little interst to us in the context of this exercise). 

```shell
Camera Initialized

interpretPrediction: OneFinger=0 TwoFingers=0 ThreeFingers=0.992041 FourFingers=0 FiveFingers=0
interpretPrediction player_move: 2
interpretPrediction: OneFinger=0 TwoFingers=0 ThreeFingers=0.992041 FourFingers=0 FiveFingers=0
interpretPrediction player_move: 2
interpretPrediction: OneFinger=0 TwoFingers=0 ThreeFingers=0.992041 FourFingers=0 FiveFingers=0
interpretPrediction player_move: 2
interpretPrediction: OneFinger=0 TwoFingers=0 ThreeFingers=0 FourFingers=0.919658 FiveFingers=0
interpretPrediction player_move: 2
interpretPrediction: OneFinger=0 TwoFingers=0 ThreeFingers=0.992041 FourFingers=0 FiveFingers=0
interpretPrediction player_move: 2
classcount:
0 0 4 1 0
---- ---- ---- ----
```

If you want to explore this part and build a different script, start by looking at the src/tinyml_deployment/main/src/PredictionHandler.cpp file...




