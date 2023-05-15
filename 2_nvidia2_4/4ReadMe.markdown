### 目录


# [NVIDIA](https://so.csdn.net/so/search?q=NVIDIA\&spm=1001.2101.3001.7020)的2:4 pattern稀疏方案

## 前言

> 手写AI推出的全新[模型剪枝](https://so.csdn.net/so/search?q=%E6%A8%A1%E5%9E%8B%E5%89%AA%E6%9E%9D\&spm=1001.2101.3001.7020)与重参课程。记录下个人学习笔记，仅供自己参考。
>
> 本次课程主要讲解NVIDIA的2:4[剪枝](https://so.csdn.net/so/search?q=%E5%89%AA%E6%9E%9D\&spm=1001.2101.3001.7020)方案。
>
> reference:
>
> ASP nvidia 2:4 pattern pruning
>
> paper:
>
> *   [Accelerating Sparse Deep Neural Networks](https://arxiv.org/pdf/2104.08378.pdf)
>
> code:
>
> *   <https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity>
>
> blog:
>
> *   <https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/>
>
> tensor core:
>
> *   <https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/>
>

## 1.稀疏性的研究现状

许多研究集中在两方面：

*   大量(80-95%)的非结构化、细粒度稀疏
*   用于简单加速的粗粒度稀疏

这些方法所面临的挑战有：

*   **精度损失**

    *   高稀疏度往往会导致准确率损失几个百分点，即使拥有先进的训练技术也是如此
*   **缺少一种适用于不同任务和网络的训练方法**

    *   恢复准确性的训练方法因网络而异，通常需要超参数搜索
*   **缺少加速**

    *   Math：非结构数据难以利用现代向量/矩阵数学指令的优势
    *   Memory access：非结构化数据往往不能很好地利用内存总线，由于读操作之间存在依赖关系，导致延迟增加
    *   Storage overheads：metadata占用的存储空间比非零权重多消耗2倍，从而抵消了一些压缩的好处。(**metadata通常指的是对于权重矩阵的稀疏性描述信息**，例如哪些位置是零元素，哪些位置是非零元素)

## 2.图解nvidia2-4稀疏方案

NVIDIA在处理稀疏矩阵W时，会采用2:4稀疏方案。在这个方案中，稀疏矩阵W**首先**会被压缩，压缩后的矩阵存储着非零的数据值，而**metadata**则存储着对应非零元素在原矩阵W中的索引信息。具体来说，metadata会将W中非零元素的行号和列号压缩成两个独立的一维数组，这两个数组就是metadata中存储的索引信息。如下图所示：

![在这里插入图片描述](https://github.com/powerdoudou/img/blob/main/Prune/lesson4/1.png?raw=true)

对于大型矩阵相乘时，我们可以采用**2:4稀疏方案减少计算量**，假设矩阵A和B相乘得到C，正常运算如下图所示：

![在这里插入图片描述](https://github.com/powerdoudou/img/blob/main/Prune/lesson4/2.png?raw=true "在这里插入图片描述")


我们可以将A矩阵进行剪枝使其变得稀疏，如下图所示：

![在这里插入图片描述](https://github.com/powerdoudou/img/blob/main/Prune/lesson4/3.png?raw=true)

而针对于稀疏矩阵，我们可以通过上述的NVIDIA方案将其变为2:4的结构，可以将A矩阵进行压缩，而对矩阵B的稀疏是通过硬件上面的Sparse Tensor Cores进行选择，如下图所示：

![在这里插入图片描述](https://github.com/powerdoudou/img/blob/main/Prune/lesson4/4.png?raw=true "在这里插入图片描述")

## 3.训练策略

NVIDIA提供的2:4稀疏训练方案步骤如下：

*   1\) 训练网络
*   2\) 2:4稀疏剪枝
*   3\) 重复原始的训练流程

    *   超参数的选择与步骤1一致
    *   权重的初始化与步骤2一致
    *   保持步骤2中的 0 patter：不需要重新计算mask

图示如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/408e0b44a78743a3a8014765870d2bac.png#pic_center "在这里插入图片描述")

## 4.手写复现

### 4.1 大体框架

#### &#x20;4.1.1 大体流程

1.  train
2.  pruned
3.  finetune & save model

#### &#x20;4.1.2 ASP类的实现

Automatic SParsity

1.  对训练好的模型进行剪枝： prune\_trained\_model
2.  对模型进行初始化 init\_model\_for\_pruning
3.  对优化器进行初始化 init\_model\_for\_pruning

剪枝方案为ASP，主要实现的是前面提到过的2:4稀疏剪枝，其具体实现细节在ASP类中。

### 4.2 ASP类的实现

ASP类的实现示例代码如下：

在上面的示例代码中，ASP的类方法`prune_trained_model`会对训练好的模型进行剪枝操作，首先它会去调用`init_model_for_pruning`和`init_optimizer_for_pruning`对模型和优化器进行初始化，然后调用`compute_sparse_masks`生成稀疏掩码(**具体实现见4.3**)，最后使用掩码对模型进行剪枝。

whitelist ：可以被剪裁的模块

m4n2\_1d: 一维里每四个选两个

```python
class ASP():
    
    @classmethod
    def init_model_for_pruning(model, mask_calculater, whitelist):
        pass

    @classmethod
    def init_optimizer_for_pruning(optimizer):
        pass
    
    @classmethod
    def compute_sparse_masks():
        pass

    @classmethod
    def prune_trained_model(cls, model, optimizer):
        cls.init_model_for_pruning(
            model,
            mask_calculater = "m4n2_1d",
            whitelist = [torch.nn.Linear, torch.nn.Conv2d]
        )
        cls.init_optimizer_for_pruning(optimizer)

        cls.compute_sparse_masks()  # 2:4
```

### 补充知识：设计模式

用下面的代码可能出现的问题：

1.每个pattern的实现内容非常复杂，代码看起很杂乱

避免以后每次拓展新的pattern都要在if后面添加函数。

希望：代码可以自动找到patter的选项,say m4n2\_1d;

```python
###此方法不好
def create_mask(weight, pattern):
	if pattern=='m4n2_1d':
		pass
	if pattern=='m4n2_2d':
		pass
	if pattern=='m6n3_1d':
		pass
	else:
		raise ValueError("Invalid pattern")
	
	return mask.bool()


```

```python
return mask.bool()

```

### &#x20;4.3 mask的实现

#### &#x20;4.3.1 NVIDIA的方案 Steps

1.  创建一个patterns，如下图所示，由于是2:4的方案，所有总共有6种不同的pattern
2.  将weight matrix变换成nx4的格式方便与pattern进行矩阵运算
3.  运算后的结果为nx6的矩阵，在n的维度上进行argmax取得最大的索引(**索引对应pattern**)，然后将索引对应的pattern值填充到mask中

`mn_1d_best`函数实现了在指定大小的矩阵中寻找最佳的m:4的mask矩阵，具体实现可看上述的图示流程，`m4n2_1d`函数则是对`mn_1d_best`函数的进一步封装，指定了m=4,n=4，即寻找最佳的2:4的mask矩阵，`create_mask`函数则根据给定的权重矩阵、mask生成函数名和稀疏度生成相应的mask矩阵。

1.  mask = create\_mask &#x20;
2.  func = getattr
3.  mask = mn\_1d\_best
4.  pattern = compute\_valid\_1d\_patterns
5.  mat=reshape\_1d

参考代码：

![在这里插入图片描述](https://img-blog.csdnimg.cn/caa2aaa44b924980827ce1b4b77a2d90.png#pic_center "在这里插入图片描述")

![在这里插入图片描述](https://img-blog.csdnimg.cn/f8013000ae9648e7961e3bef0434c10d.png#pic_center "在这里插入图片描述")

![在这里插入图片描述](https://img-blog.csdnimg.cn/4a115d7e282740eaab793ea295ff0967.png#pic_center "在这里插入图片描述")

```python
torch.Tensor
```

可视化结果如下，其中mask中白色区域填充的是0，黑色区域代表的是1：

![在这里插入图片描述](https://img-blog.csdnimg.cn/955b9c87f46f46ed89b6cdd3375bc8d4.jpeg#pic_center "在这里插入图片描述")

### 4.4 模型初始化
