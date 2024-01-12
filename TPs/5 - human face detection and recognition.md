# ESP WHO Face detection and recognition

One reason the esp32 board  is commonly used with AIML is because of its 2 core CPUs, and thus its ability to run neural networks. Combined with a camera and a microphone, this setup makes for many interesting possibilities.

This does not mean that running a neural network is straighforward. The model structure, size, and its quantization, make all the difference between a small model that runs well and a bigger one that hangs of outputs low accuracy. Espressif has developped examples that were fine tuned to showcase how easy NNs run on their board. They spent a lot of time tuning the code to make it fast, and make it seem that any neural neural network would run fast.

Let's explore their implementation, then see if we can build a customized code.

This lab supposed that you have already cloned the ESP-WHO git, if not, position yourself in a directory (e.g., ~/IMT/esp-idf) and clone their git repo:

```shell
$ git clone -b idfv4.4 --recursive https://github.com/espressif/esp-who.git
```

Once the install completes, navigate to the esp-who example folder:

```shell
$ cd esp-who/examples 
```

Use the 'ls' command to check the content. You will see different subdirectories. Each example includes different subfolders, with the calls required to compile the example for the terminal, LCD or web interfaces. Not all platforms support all interfaces. Navigate to the Human face detection web interface folder (the face recognition example is a dependency of the face detection):

```shell
$ cd human_face_detection/web
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

From your laptop, look for the available list of Wi-FI SSIDs. You should see one named human face detection. Connect to that SSID. The SSID runs on the ESP32, and will provide to your laptop an IP address (192.168.4.2). Open a web browser to your ESP32 web server (at 192.168.4.1). You should see the main interface. Activate face detection, start the stream, and position a face in front of the camera (50 cm to 1 m seems to be a good range). Make sure that there is enough contrast between the face and the background, and good lighting. The system should detect a face, and display the detection on the CLI, but also on the web interface (showing the face with a green square around, and the eyes and other face features marked with dots).

At the bottom, you can also activate face recognition. Position your face in front of the camera and click Enroll face. You can register your user, and the NN should be able to flash the user when the correct face is in front of the camera. You can also use the Boot button on the development board for interaction:

* Short press the button: recognize the face captured by the camera in real time.
* Long press the button: enroll the face captured by the camera in real time.
* Double click the button: delete the last enrolled face.


## What happens in the background? Looking at the tflite implementation

The Espressif model is difficult to dissect, because it is compiled optimized. An earlier version gives us better visibility into the code, as Espressif was working on optimizing (and realising) an esp-optimized set of [neural networks](https://github.com/espressif/esp-nn#performance) derived from tflite libraries and models. They worked from [tflite-micro](https://github.com/tensorflow/tflite-micro), which is a great place to start if you want to dig deeper into implementations for ESP. Tis earleir version is found in the esp-tflite-micro repository. You should have installed it in a previous exercise. If you lost it or forgot, start by cloning their repository from your root directory (e.g., ~/IMT/esp-idf/):

```shell
$ git clone --recursive https://github.com/espressif/esp-tflite-micro.git
```

Then, in the folder, and example sub-folder, you will see 3 examples. Navigate to person_detection:

```shell
$ cd esp-tflite-micro/examples/person_detection/ 
```

As usual, the README.md file in the folder (also [here](https://github.com/espressif/esp-tflite-micro/tree/master/examples/person_detection)) provides an intro that explains some of the basics of what they try to do. 
In general, the goal of the tool is to recognize a human face (vs. an animal face or anything else). This is a classifier with two categories ('human', 'other'). A CNN is a good tool for that detection, because it can analyze human images and find patterns that tend to come often (like eyes, eyebrows etc.), including in comparison to non-human faces (cats, for example, tend to have eyes, but also pointy things on their heads that humans often lack).

The implementation is a bit different from the WHO later version. Try this one. Set your ESP-EYE as the target:

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

Start showing faces and others to the ESP-EYE, each time the tool will detect a person or something else, it should display the probabilities on the terminal:

```shell
person score:27%, no person score 73%
Image Captured

person score:95%, no person score 5%
Image Captured
```

The UI details (web interface or CLI, with probability score or yes/no) are not very important. What is more interesting for us is the AIML side. The two key files are in the 'main' folder, namely person_detect_model_data.h and person_detect_model_data.cc.
Look at person_detect_model_data.h first. You will see that the file comment section at the begining explains that the mode was converted to a C data array  with an xxd command, from the person_detect.tflite file, to the person_detect_model_data.cc file. Great, so the C array is the second file. It would be intersting to look at the person_detect.tflite model. Unfortunately, it is not provided. But we can work from the cc file and convert it back to a tflite file!
Start by making a copy of person_detect_model_data.cc, and changing it to txt:

```shell
$ cp person_detect_model_data.cc person_detect_model_data.txt
```

Then, move person_detect_model_data.txt to the folder where you run your Jupyter notebooks.

Next, we can load this file and convert it back to tflite. There are a few ways about, the below is just one of them. One way is to remove anything that is not the model from the file. You can use vi to do so (remember, in vi, 'dd' allows you to delete a line, 'G' allows you to jump to then end of the file, and 'i' allows you interactive editing, for example to delete a comma. Use these commands to clean the model file. The file should then only contain the mode, i.e., the first line should start with 0x1c, 0x00, ... and the last line should end with 0x02, 0x00, 0x00, 0x00

Then, you can use python to load that file and convert it to tflite:

```shell
def parse_model_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        # Assuming the file content is in the format: 0x2d, 0x1f, 0x00, ...
        hex_values = content.split(',')
        byte_values = [int(hv.strip(), 16) for hv in hex_values]
        return byte_values

# Replace 'path_to_your_model_file' with the actual file path
model_data = parse_model_file('person_detect_model_data.txt')

# Convert to bytes and write to a .tflite file
model_bytes = bytes(model_data)
with open("person_detect.tflite", "wb") as file:
    file.write(model_bytes)
```

You should now have the person_detect.tflite file. Use the tools you learned (netron, or tensorflow API) to look at the structure of the file. You should see 28 layers, using Conv2D, depthwiseConv2D and relu. It is interesting to observe the number of units for each layer. You will see that the increase as the neural network gets forward. This structure is classical of optimization tools (e.g., Optuna) that attempt to find the best structure for a given accuracy target. These tools start by finding an optimal number of layers, then walk backward from the output layer to try to increase accuracy by increasing the size of each layer (then they stop when reaching an acceptable accuracy target).

One last item is unclear: the input (what kind of image does the model take as input)? You see in the model 1x96x96x1, which seems to imply one image at a time, 96x96 grayscale. Luckily, the developpers of the demo also thought that you may be on a system with no camera, wehre you would then want to test the neural engine without the ability to show an actual image. To help this case, they included a static_images/sample_images folder, with 10 images coming from the training set (if you are curious, the README.md file in that folder explains how they expect you to use these images).

Try to open these images. You may have an app on your computer that can read them. If not, go to https://rawpixels.net, and upload the images one by one. You will see their format.









