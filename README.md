# Object Detection
## Object Detection using the TensorFlow Object Detection API

Forked from: https://www.freecodecamp.org/news/how-to-play-quidditch-using-the-tensorflow-object-detection-api-b0742b99065d/

### Getting started
Start by cloning my GitHub repository or can clone the TensorFlow models [repo](https://github.com/tensorflow/models). If you choose the latter, you only need the folders named "slim" and "object_detection", so feel free to remove the rest. Don't rename anything inside these folders (unless you're sure it won't mess with the code).
```bash
git clone git@github.com:datts68/object_detection.git
```

Start Tensorflow container:
```bash
docker run --gpus all -it --mount type=bind,source="$(pwd)"/object_detection,target=/app tensorflow/tensorflow:1.15.2-gpu bash
```

### Dependencies
Assuming you have TensorFlow installed, you may need to install a few more dependencies, which you can do by executing the following in the base directory:
```bash
pip install -r requirements.txt
```

The API uses Protobufs to configure and train model parameters. We need to compile the Protobuf libraries before using them. First, you have to install the Protobuf Compiler using the below command:
```bash
sudo apt-get install protobuf-compiler
```

Now, you can compile the Protobuf libraries using the following command:
```bash
protoc object_detection/protos/*.proto --python_out=.
```

You need to append the path of your base directory, as well as your slim directory to your Python path variable. Note that you have to complete this step every time you open a new terminal. You can do so by executing the below command. Alternatively, you can add it to your ~/.bashrc file to automate the process.
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### Preparing the inputs
We start by preparing the **label_map.pbtxt** file. This would contain all the target label names as well as an ID number for each label. Note that the label ID should start from 1. Here’s the content of the file that I used for my project.
```bash
item {
  id: 1
  name: 'snitch'
}

item {
  id: 2
  name: 'quaffle'
}

item {
  id: 3
  name: 'bludger'
}
```
Now, its time to collect the dataset.

Fun! Or boring, depending on your taste, but it's a mundane task all the same.

I collected the dataset by sampling all the frames from a Harry Potter video clip, using a small code snippet I wrote, using the OpenCV framework. Once that was done, I used another code snippet to randomly sample 300 images from the dataset. The code snippets are available in **utils.py** in my GitHub [repo](https://github.com/datts68/object_detection/blob/master/utils.py) if you would like to do the same.

You heard me right. Only 300 images. Yeah, my dataset wasn't huge. That's mainly because I can't afford to annotate a lot of images. If you want, you can opt for paid services like Amazon Mechanical Turk to annotate your images.

### Annotations
Every image localization task requires ground truth annotations. The annotations used here are XML files with 4 coordinates representing the location of the bounding box surrounding an object, and its label. We use the Pascal VOC format. A sample annotation would look like this:
```bash
<annotation verified="no">
  <filename>3.jpg</filename>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>1280</width>
    <height>586</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>quaffle</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <Difficult>0</Difficult>
    <bndbox>
      <xmin>458</xmin>
      <ymin>88</ymin>
      <xmax>490</xmax>
      <ymax>118</ymax>
    </bndbox>
  </object>
</annotation>
```
You might be thinking, "Do I really need to go through the pain of manually typing in annotations in XML files?" Absolutely not! There are tools which let you use a GUI to draw boxes over objects and annotate them. Fun! [LabelImg](https://github.com/tzutalin/labelImg) is an excellent tool for Linux/Windows users. Alternatively, **RectLabel** is a good choice for Mac users.

A few footnotes before you start collecting your dataset:
- Do not rename you image files after you annotate them. The code tries to look up an image using the file name specified inside your XML file (Which LabelImg automatically fills in with the image file name). Also, make sure your **image** and **XML** files have the **same name**.
- Make sure you **resize** the images to the desired size **before** you start annotating them. If you do so later on, the annotations will not make sense, and you will have to scale the annotation values inside the XMLs.
- LabelImg may output some extra elements to the XML file (Such as <pose>, <truncated>, <path>). You do not need to remove those as they won’t interfere with the code.

In case you messed up anything, the **utils.py** file has some utility functions that can help you out. If you just want to give Quidditch a shot, you could download my annotated dataset instead. Both are available in my GitHub repository.

Lastly, create a text file named **trainval**. It should contain the names of all your image/XML files. For instance, if you have img1.jpg, img2.jpg and img1.xml, img2.xml in your dataset, you trainval.txt file should look like this:
```bash
img1
img2
img3
...
```

Separate your dataset into two folders, namely **images** and **annotations**. Place the **label_map.pbtxt** and **trainval.txt** inside your annotations folder. Create a folder named xmls inside the annotations folder and place all your XMLs inside that. Your directory hierarchy should look something like this:
```bash
base_directory
|_images
|_annotations
  |_xmls
  |_label_map.pbtxt
  |_trainval.txt
```

The API accepts inputs in the **TFRecords** file format. Worry not, you can easily convert your current dataset into the required format with the help of a small utility function. Use the **create_tf_record.py** file provided in my repo to convert your dataset into TFRecords. You should execute the following command in your base directory:
```bash
python create_tf_record.py --data_dir=`pwd` --output_dir=`pwd`
```

You will find two files, **train.record** and **val.record**, after the program finishes its execution. The standard dataset split is 70% for training and 30% for validation. You can change the split fraction in the main() function of the file if needed.

### Training the model
Whew, that was a rather long process to get things ready. The end is almost near. We need to select a localization model to train. Problem is, there are so many options to choose from. Each vary in performance in terms of speed or accuracy. You have to choose the right model for the right job. If you wish to learn more about the trade-off, this [paper](https://arxiv.org/abs/1611.10012) is a good read.

In short, SSDs are fast but may fail to detect smaller objects with decent accuracy, whereas Faster RCNNs are relatively slower and larger, but have better accuracy.

The TensorFlow Object Detection API has provided us with a bunch of [pre-trained models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). It is highly recommended to initialize training using a pre-trained model. It can heavily reduce the training time.

Download one of these models, and extract the contents into your base directory. Since I was more focused on the accuracy, but also wanted a reasonable execution time, I chose the ResNet-50 version of the Faster RCNN model. After extraction, you will receive the model checkpoints, a frozen inference graph, and a pipeline.config file.

One last thing remains! You have to define the “training job” in the pipeline.config file. Place the file in the base directory. What really matters is the last few lines of the file — you only need to set the highlighted values to your respective file locations.
```bash
train_config {
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "model.ckpt"
  num_steps: 200000
}
train_input_reader {
  label_map_path: "annotations/label_map.pbtxt"
  tf_record_input_reader {
    input_path: train.record"
  }
}
eval_input_reader {
  label_map_path: "annotations/label_map.pbtxt"
  tf_record_input_reader {
    input_path: val.record"
  }
}
```

If you have experience in setting the best hyper parameters for your model, you may do so. The creators have given some rather brief guidelines [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md).

You’re all set to train your model now! Execute the below command to start the training job.
```bash
python object_detection/train.py --logtostderr --pipeline_config_path=pipeline.config --train_dir=train
```

You can resume training from a checkpoint by modifying the **fine_tune_checkpoint** attribute from model.ckpt to model.ckpt-xxxx, where xxxx represents the global step number of the saved checkpoint.

### Exporting the model for inference
What's the point of training the model if you can't use it for object detection? API to the rescue again! But there's a catch. Their inference module requires a frozen graph model as an input. Not to worry though: using the following command, you can export your trained model to a frozen graph model.
```bash
python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=pipeline.config --trained_checkpoint_prefix=train/model.ckpt-xxxxx --output_directory=output
```

Neat! You will obtain a file named frozen_inference_graph.pb, along with a bunch of checkpoint files.

You can use it to test or run your object detection module. The code is pretty self explanatory, and is similar to the Object Detection Demo, presented by the creators. You can execute it by typing in the following command:
```bash
python object_detection/inference.py --input_dir={PATH} --output_dir={PATH} --label_map={PATH} --frozen_graph={PATH} --num_output_classes={NUM}
```

Replace the highlighted characters **{PATH}** with the filename or path of the respective file/directory. Replace **{NUM}** with the number of objects you have defined for your model to detect (In my case, 3).

### Results
Check out these videos to see its performance for yourself! The first video demonstrates the model’s capability to distinguish all three objects, whereas the second video flaunts its prowess as a seeker.

Pretty impressive I would say! It does have an issue with distinguishing heads from Quidditch objects. But considering the size of our dataset, the performance is pretty good.

Training it for too long led to massive over-fitting (it was no longer size invariant), even though it reduced some mistakes. You can overcome this by having a larger dataset.
