# Pytorch Image Classifier Template
This code modifies the output layer of image classification network commonly used in pytorch. The modified model can be used to process any number of image classification data. At the same time, the pre-training parameters and re-training parameters are distinguished for fine-tuning and component training of parameters. We have modified the following network structure:
<table border="1">
    <tr>
        <td>MobileNet</td>
        <td>mobilenet_v2</td>
        <td>mobilenet_v3_small</td>
        <td>mobilenet_v3_large</td>
    </tr>
    <tr>
        <td>ResNet</td>
        <td>resnet18</td>
        <td>resnet34</td>
        <td>resnet50</td>
        <td>resnet101</td>
    </tr>
    <tr>
		<td>ResNeXt</td>
        <td>resnext50_32x4d</td>
        <td>resnext101_32x8d</td>
    </tr>
    <tr>
		<td>DenseNet</td>
        <td>densenet121</td>
        <td>densenet161</td>
        <td>densenet169</td>
    </tr>
    <tr>
		<td>ShuffleNet</td>
        <td>shufflenet_v2_x0_5</td>
        <td>shufflenet_v2_x1_0</td>
    </tr>
    <tr>
		<td>SqueezeNet</td>
        <td>squeezenet1_0</td>
        <td>squeezenet1_1</td>
    </tr>
    <tr>
		<td>WideResNet</td>
        <td>wide_resnet50_2</td>
        <td>wide_resnet101_2</td>
    </tr>
</table>

# Blog
* [从零搭建音乐识别系统（一）整体功能介绍](https://blog.csdn.net/cxx654/article/details/118254718?spm=1001.2014.3001.5501)

# Usage
* modify data/classifier.yaml
    - classes: class list
    - dataset_dir: dataset path
    - save_dir: model save path
    - train_size: sample number per class
* model train: `python train_classifier_model.py`
* model inference: `python model_inference.py`