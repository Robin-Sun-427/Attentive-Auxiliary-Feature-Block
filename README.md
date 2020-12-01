# Attentive-Auxiliary-Feature-Block


The personal implement of paper《Lightweight Single-Image Image SR with Attentive Auxiliary Feature Learning》(ACCV) by Pytorch.
---------

    I will keep updated

## 2020 11/25 updated!

    (1)implement the model part and the model base part
 * 完成了model和model_base 部分


## 2020 11/27 updated!

 (1)`main.py` for the trainning phase;trainning set used public datasets BSDS300.
 * 完成了训练过程，训练数据集使用的是BSDS300,当然您可以更换

 (2)`predict_SR.py` for the testing a image.
 * 完成了使用测试集的测试过程
    

## 2020 12/1 updated!

 (1)use image patches pairs created by Random Crop approach as network model input datasets 
    
 * 更新`datasets.py`：对网络输入图像进行随机裁剪，构成patches pairs 作为input和label



