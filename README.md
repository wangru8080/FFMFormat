# FFMFormat

FM和FFM模型是最近几年提出的模型，凭借其在数据量比较大并且特征稀疏的情况下，仍然能够得到优秀的性能和效果的特性，常用于计算广告中的CTR，CVR预估。美团点评技术团队写过一篇《深入FFM原理与实践》博客，链接地址：https://tech.meituan.com/deep-understanding-of-ffm-principles-and-practices.html ，写的很详细。然后目前FFM常用的python库有libffm，xlearn等。

## Data Format(libffm格式)

为了使用FFM方法，所有特征必须转化成 

label field_id:feature_id:value field_id:feature_id:value field_id:feature_id:value...的格式。

field_id表示每一个特征域的id号

feature_id表示所有特征值的id号（可采用连续编码以及hash编码）

value：当特征域不是连续特征时，value=1，若为连续特征，value=该特征的值

## libffm格式

有必要理解什么是field，feature,value。举个例子：

对于pandas DataFrame格式数据来说:

    label  category_feature  continuous_feature  vector_feature
    =====  ================  ==================  ==============
    0           x               1.1               1 2
    1           y               1.2               3 4 5   
    0           x               2.2               6 7 8 9
    

### 首先给各个特征域编号：

dict_field[category_feature] = 0

dict_field[continuous_feature] = 1

dict_field[vector_feature] = 2

### 然后再给每个特征值编号：

dict_feature[category_feature-x] = 0

dict_feature[continuous_feature-1.1] = 1

dict_feature[vector_feature-1] = 2

dict_feature[vector_feature-2] = 3

dict_feature[category_feature-y] = 4

dict_feature[continuous_feature-1.2] = 5

dict_feature[vector_feature-3] = 6

dict_feature[vector_feature-4] = 7

dict_feature[vector_feature-5] = 8

dict_feature[category_feature-x] = 0 # category_feature重复项编号不变

dict_feature[continuous_feature-2.2] = 9

dict_feature[vector_feature-6] = 10

dict_feature[vector_feature-7] = 11

dict_feature[vector_feature-8] = 12

dict_feature[vector_feature-9] = 13


### 最后设置value的值：

dict_value[category_feature-x] = 1

dict_value[continuous_feature-1.1] = 1

dict_value[vector_feature-1] = 1

dict_value[vector_feature-2] = 1

dict_value[category_feature-y] = 1

dict_value[continuous_feature-1.2] = 1.2

dict_value[vector_feature-3] = 1

dict_value[vector_feature-4] = 1

dict_value[vector_feature-5] = 1

dict_value[category_feature-x] = 1 

dict_value[continuous_feature-2.2] = 2.2

dict_value[vector_feature-6] = 1

dict_value[vector_feature-7] = 1

dict_value[vector_feature-8] = 1

dict_value[vector_feature-9] = 1


综上，我们可得到FFM Format data：

    0 0:0:1 1:1:1.1 2:2:1 2:3:1

    1 0:4:1 1:5:1.2 2:6:1 2:7:1 2:8:1

    0 0:0:1 1:9:2.2 2:10:1 2:11:1 2:12:1 2:13:1


本文仅有category_feature，continuous_feature，vector_feature。若还有其他特征可自行修改添加。

libffm库参考：https://github.com/guestwalk/libffm

xlearn使用方法参考：http://xlearn-doc.readthedocs.io/en/latest/start.html

