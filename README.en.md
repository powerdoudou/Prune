# LearnPrune

#### Description
Includes the basics of pruning vgg,yolov8 aspects

#### Software Architecture
Software architecture description

#### Remove

1.  Build Bigmodel, register buffer, train and export origin onnx
2.  Export the model to ONNX format
3.  Prune model with norm1 value
4.  Train pruned model

#### KeyPoints

1.  self.conv1_l1norm = nn.Parameter(torch.Tensor(32), requires_grad=False)
2.  self.register_buffer('conv1_l1norm_buffer', self.conv1_l1norm)
3.  self.conv1_l1norm.data=torch.sum(torch.abs(self.conv1.weight.data),dim=(1,2,3))
        

#### Prune Steps
1. load a model and inspect it
2. get the global threshold according to the l1norm
3. prune the conv based on the l1norm along axis = 0 for each weight tensor
    Top conv

    
![图片](fig\tb_conv_pruning.jpg)
    Bottom conv
![prune](fig\16_prune.png)

4. Save the pruned model state_dict
5. Set the input shape of the model
6. Export the model to ONNX format

FINE TUNE

7. Prepare the MNIST dataset
8. Save the trained big network

#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
