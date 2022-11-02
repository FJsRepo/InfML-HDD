<div align="center">

# HorizonNet
![HorizonNet](figures/Flowchart.png "HorizonNet")
</div>

## About this Algorithm
The proposed algorithm is designed for horizon detection.
This code aims to reproduce the results of in our paper, hence part of the test data is included in ./data/test_set and 
the trained model from model_386 to model_416 can be downloaded [here](https://drive.google.com/drive/folders/1Vt3C1QYJJp4FzUwde7bj444rH5MgxEIv?usp=sharing), and after download, you should decompression them in ./experiments.

The framework of the project is based on [PolyLaneNet](https://github.com/lucastabelini/PolyLaneNet) from Lucas Tabelini for Lane detection (thanks for their outstanding work),
and changes have been made according to the characteristics and application fields of horizon detection.
 
The main innovation point is that a new network structure was designed, which deeply integrate the the thoughts of traditional hand-crafted features and CNN to improve detection efficiency.

Other contributions：

(1)An HLL based on the HLF is designed and optimized specially to process the feature maps to highlight the target and eliminate the interferences simultaneously. We carefully studied the principle of the HLF and proved its effectiveness through extensive experiments.

(2)According to the structural characteristics of the multiple feature maps inside CNN, a Positioning-Module is presented, which can quickly detect the position of the horizon with acceptable additional computational cost. 

(3)By integrating the traditional idea modules deeply into CNN, the proposed strategy integrates both the intuitiveness of the traditional method and the powerful feature extraction capability of CNN. Moreover, to fully estimate the performance of the algorithm proposed in this paper, a self-built dataset named HorizonSet with more than 6000 samples that covers a variety of complex scenes is presented.

## About the dataset
HorizonSet was constructed according to the structure of the Tusimple dataset, for every horizon line, 13 gt points were provided.  
You could acquire the marked images by yourself with the code in ./nets/Proposed.py and test.py via the annotated lines.

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