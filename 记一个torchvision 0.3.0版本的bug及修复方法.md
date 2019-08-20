### 写在前面
本文描述的`bug`在`torchvision 0.4.0`版本已经修复，事实上正是通过对比`0.3.0`版本与`0.4.0`版本的差异才确定的`bug`；<br/>
而由于`torch 1.2.0 torchvision 0.4.0` 不支持`cuda 9.0`，而又懒得升级`cuda`，才想方设法进行修复

### 情况环境
* CUDA 9.0
* torch 1.1.0
* torchvision 0.3.0

### 使用torchvision下detection模型是报错
使用预训练好的模型进行目标检测，不管是`maskrcnn`还是`fasterrcnn`，详见[TorchExample](https://github.com/penchy-zju/code_gist/blob/master/torch_example.py)当输入图片不是默认`800*800`时就会报错如下：
```
  File "maskrcnn.py", line 45, in generate
    p = mask_rcnn(inputBatch)
  File "/home/appops/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/appops/anaconda3/lib/python3.6/site-packages/torchvision/models/detection/generalized_rcnn.py", line 47, in forward
    images, targets = self.transform(images, targets)
  File "/home/appops/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/appops/anaconda3/lib/python3.6/site-packages/torchvision/models/detection/transform.py", line 41, in forward
    images[i] = image
RuntimeError: The expanded size of the tensor (1067) must match the existing size (800) at non-singleton dimension 2.  Target sizes: [3, 1067, 1067].  Tensor sizes: [3, 800, 800]
```
从上面的报错信息可以看到出错的地方是`torchvision/models/detection/transform.py`，经过对比`torchvision 0.3.0` 和 `0.4.0` 两个版本文件可以发现，
在`0.4.0`版本中在数据预处理的时候对tensor进行了拷贝，这样图片缩放就不会影响原来的输入图片；
``` python
def forward(self, images, targets=None):
    images = [img for img in images] # 0.3.0版本没有这一行
    for i in range(len(images)):
        image = images[i]
        target = targets[i] if targets is not None else targets
        if image.dim() != 3:
            raise ValueError("images is expected to be a list of 3d tensors "
                             "of shape [C, H, W], got {}".format(image.shape))
        image = self.normalize(image)
        image, target = self.resize(image, target)
        images[i] = image
        if targets is not None:
            targets[i] = target
    image_sizes = [img.shape[-2:] for img in images]
    images = self.batch_images(images)
    image_list = ImageList(images, image_sizes)
    return image_list, targets
```

### 修复方法
最好的修复方法其实是升级版本，`0.4.0`已经对bug进行了修复，但由于新版本`cuda`支持的问题，我尝试使用笨办法修复：

直接打开安装包里的文件(我这里安装好后文件位置是`anaconda3/lib/python3.6/site-packages/torchvision/models/detection/transform.py`)，加上上面那句代码，重试，OK！
