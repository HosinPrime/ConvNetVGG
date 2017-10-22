各文件解释:
data:
	里面为训练和测试数据
	每个里面包括不同的文件夹，分别为不同类型的图片数据
Process.py
	进行数据的处理

tools.py
	定义了一席所需要的卷积，池化等等数据操作
Vgg.py
	定义了vgg模型
train.py
	进行训练

要使用的话直接在train里面设定好自己的训练数据和测试数据路径，log路径,然后再该readme.txt所在的
路径运行python train.py就可以进行模型的训练了
