## 天朝客运量预测实验

本次练习使用TensorFlow实现，在RussellCloud平台运行，感谢斗大的熊猫提供参考博客。

### 时间序列问题简介

之前的一篇文章「如何检测业务数据中的异常」以及时间序列的一些文章中有专门讲过时间序列方面的一些常规处理方式，简单来说，时间序列预测分析就是利用过去一段时间内某事件时间的特征来预测未来一段时间内该事件的特征。

### 时间序列与RNN

时间序列模型最常用最强大的的工具就是递归神经网络（recurrent neural network, RNN）。相比与普通神经网络的各计算结果之间相互独立的特点，RNN的每一次隐含层的计算结果都与当前输入以及上一次的隐含层结果相关。通过这种方法，RNN的计算结果便具备了记忆之前几次结果的特点。

典型的RNN网路结构如下：

![](https://pic3.zhimg.com/50/v2-d7ce9d95210cbc05eebaebcebb97b166_hd.jpg)

### RNN的局限：
由于RNN模型如果需要实现长期记忆的话需要将当前的隐含态的计算与前n次的计算挂钩，即St = f(U*Xt + W1*St-1 + W2*St-2 + ... + Wn*St-n)，那样的话计算量会呈指数式增长，导致模型训练的时间大幅增加，因此RNN模型一般直接用来进行长期记忆计算。

LSTM模型
LSTM（Long Short-Term Memory）模型是一种RNN的变型，LSTM的特点就是在RNN结构以外添加了各层的阀门节点。阀门有3类：遗忘阀门（forget gate），输入阀门（input gate）和输出阀门（output gate）。这些阀门可以打开或关闭，用于将判断模型网络的记忆态（之前网络的状态）在该层输出的结果是否达到阈值从而加入到当前该层的计算中。以此来解决RNN的一部分弊端。


数据集截图：

![](https://pic4.zhimg.com/50/v2-2048e61f33c8166f23324e8cff34b31f_hd.jpg)

预测结果：

![](https://pic4.zhimg.com/50/v2-8baaca09deb0f9d06e57266beb16f2b7_hd.jpg)


复现过程：

前提：

* 使用邀请码在官网进行注册
* 在本地安装 cli 客户端
* 在cli客户端通过login登录


第一步：在官网创建项目

点击 [项目创建页](http://russellcloud.com/project/create) 创建项目，默认环境选：keras (注意不要选 kera:py2 )

![](https://pic1.zhimg.com/50/v2-d2a423b86456f3f77cc17bfcb8e16b78_hd.jpg)


第二步：绑定本地项目


```
#clone项目代码
git clone https://github.com/RussellCloud/train_predict.git


#进入项目目录
cd train_predict

#通过项目名初始化项目
russell init --name train_passenger_predict
```

第二步：启动notebook任务

注意--data后面接的是数据集具体版本的ID（不是数据集ID），由于这里使用的是公开数据集，所以复现时不需要替换。

```
#以jupyter模式启动
russell run --mode jupyter --data e587789c976343639d810365eb14c46c

```
启动成功后从浏览器打开返回的notebook链接即可进入相应环境



第三步：在notebook中运行代码

这一步就比较简单了，在先运行model_train训练好模型

![](https://pic2.zhimg.com/50/v2-38746badb7770a70701b6fd5af882099_hd.jpg)


，然后运行model_predict即可获得预测结果。

![](https://pic4.zhimg.com/50/v2-97f58db49843fea05bae7bb183980e63_hd.jpg)


最后一步：停止任务

切记：用完jupyter notebook以后一定要记得在终端手动stop任务，jupyter任务无法自行关闭，除非到达最大任务时间（普通用户是6小时，VIP用户是48小时）
```
# <RUN_ID>是run时返回的任务ID，也可以在网站的任务列表里获取
$ russell stop <RUN_ID>
Experiment shutdown request submitted. Check status to confirm shutdown
```

关闭后你可以在数据集页看到输出的数据集模型（因为train的时候会把model输出到output目录下，任务结束时会自动导出output目录下的文件）

![](https://pic2.zhimg.com/50/v2-00a1ad8518ee2791ea5b7a6f11df83c1_hd.jpg)