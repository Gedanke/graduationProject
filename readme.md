# graduation project

Some codes of graduation project

## Project Structure

* [readme.md](readme.md) ```readme```
* [data](data) ```dataset```
    * [originalDataSet](data/originalDataSet) ```original dataSet```
        * [car](data/originalDataSet/car)
            * [car.txt](data/originalDataSet/car/car.txt)
            * [car.names](data/originalDataSet/car/car.names)
            * [car.csv](data/originalDataSet/car/car.csv)
        * ...
    * [processedDataSet](data/processedDataSet) ```processed dataSet```
        * [car](data/processedDataSet/car)
            * [car_supervised.csv](data/processedDataSet/car/car_supervised.csv)
            * [car_unSupervised.csv](data/processedDataSet/car/car_unSupervised.csv)
            * [kMeans](data/processedDataSet/car/kMeans)
                * [car_supervised.csv](data/processedDataSet/car/kMeans/car_supervised.csv)
                * [car_unSupervised.csv](data/processedDataSet/car/kMeans/car_unSupervised.csv)
        * ...
    * [dealData](data/dealData) ```processing data set```
        * [car.py](data/dealData/car.py)
        * ...
* [core](core) ```The core code of all algorithms```
    * [basic.py](core/basic.py)
    * [core/dealData.py](core/dealData.py)
    * [divide.py](core/divide.py)
    * [model.py](core/model.py)
    * [reliefF.py](core/reliefF.py)
* [unittestMine](unittestMine) ```test```
    * ...
* [runData](runData) ```Run on each data set```
    * [car.py](runData/car.py)
    * ..
* [analysisResult](analysisResult) ```analysis the result```
    * [abalone_k.m](analysisResult/abalone_k.m)
    * [contrac_k.m](analysisResult/contrac_k.m)
    * [dermatology_k.m](analysisResult/dermatology_k.m)
    * [wall_follwing_k.m](analysisResult/wall_follwing_k.m)
* [GUI](GUI) ```The relevant code of the GUI part```
    * [resource](GUI/resource) 资源
    * ...
* [start.py](start.py) ```T entrance to the GUI```

大致流程图:

![流程图](流程图.png)

---

## 数据集

含混合特征的数据集，有标签，无标签均可，为了统一起见，都选择有标签的数据集

连续型特征需要归一化，离散特征分为有序离散特征和无序离散特征，这里都当成无序离散特征

如果缺失的样本较少，直接去除，否则对其进行一定的处理

以一定的比例  "去除" 标签，要多

处理后的文件为 csv 格式

如何确定离散特征的方差，同时使其与连续型特征的维度保持一致

小结：这部分的任务对应着 ```data``` 包，即将数据进行一定的处理，去除大部分的标签，得到符合要求的数据集，最终的每个数据集应该有两份，一份处理好的 “有监督” 含有所有标签的数据集，一份是处理好的 “无监督” 含少量标签的数据集

---

## 特征处理

连续特征需要归一化，离散特征需要有方差

方法：

熵









