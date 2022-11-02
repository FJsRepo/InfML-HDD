<div align="center">

# HorizonNet
![HorizonNet](figures/Flowchart.pdf "HorizonNet")
</div>

## About this HorizonNet
HorizonNet is designed for horizon detection.
This code aims to reproduce the results of HorizonNet in our paper, hence part of the test data is included in ./data/test_set and 
the trained model from model_386 to model_416 can be downloaded [here](https://drive.google.com/drive/folders/1Vt3C1QYJJp4FzUwde7bj444rH5MgxEIv?usp=sharing), and after download, you should put them in ./experiments/Horizon/models/.

The framework of the project is based on [PolyLaneNet](https://github.com/lucastabelini/PolyLaneNet) from Lucas Tabelini for Lane detection (thanks for their outstanding work),
and changes have been made according to the characteristics and application fields of HorizonSet.
 
The main innovation point is that a new network structure was designed, which deeply fused the CNN and the thoughts of traditional hand-crafted features to improve detection efficiency.

Other contributions：

(1)Horizon detection attempts that are completely based on CNN structures.

(2)Self-built dataset named HorizonSet, which contains more than 6000 images with six sub-datasets under different scenes.

...

## About the dataset
HorizonSet was constructed according to the structure of the Tusimple dataset, for every horizon line, 13 gt points were provided.  
Part of the stage images processed by Prior-Branch and marked tested images are listed [here](https://drive.google.com/drive/folders/142CuEn3hGg2kOixNX4qcallDWM_VRgNX?usp=sharing), they represent the binarized 
feature maps, their corresponding average row grayscale images, and the test images that are marked with gt and pred, respectively, you could acquire these images by yourself with the code in ./nets/Proposed.py and test.py via the annotated lines.

## Run test.py to validate our model
To run the code you should first install the correct environment.

The whole project is based on Python 3 and PyTorch, the version we used are Python-3.6.13 and PyTorch-1.7.1, 
the project and other key packages could be downloaded and installed by:

```
$ git clone https://github.com/FJsRepo/HorizonNet
$ cd HorizonNet
$ pip install -r requirements.txt
```

If your IDE is Pycharm, open the "Edit configuration" and set the parameters as follows:
```
--exp_name Horizon --cfg HorizonNet_test.yaml
```
![Setting](figures/Setting.jpg "Setting")

And then you can run test.py through the YAML configuration file right in the folder of HorizonNet and wait for the results.

