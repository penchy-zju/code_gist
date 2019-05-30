# 一种“奇怪”的Selective Search使用姿势

`Selective Search`（`SS`）在两阶段的目标检测算法中得到了广泛的应用，比如使用`R-CNN`就使用`SS`算法首先进行`Region Proposal`，本文将针对SS的特点介绍一种“奇怪”的使用姿势，将其应用于banner氛围图的自动裁剪中。

## SS是干嘛用的？
熟悉计算机目标检测任务的同学应该都清楚，目标检测不仅需要标识出图像中包含了什么（`classification`），还需要将目标定位（`detection`），随着深度学习的发展，目标检测任务也得到了很大的发展，就算法框架来说总体可以分为两阶段和单阶段这两类型算法；两阶段算法的第一阶段主要使用`Region Proposal`算法生成目标候选集，而SS算法就是这一类算法的典型代表；

`SS`作为一种`Region Proposal`算法，会针对输入图像给出很多可能包含物体的候选框，这些框之间可能会存在互相重叠，也有很多框本身并不包含物体，只是噪声；即便如此，这些候选框的生成也遵循着一个规律：这些生成的候选框有**较大的概率包含物体**；在目标检测的两阶段算法中，这些候选框会被一一送入分类器中进行分类，最终，超过一定概率包含物体的候选框将作为检测物体的位置结果输出。那么为了得到这一结果，就需要`Region Proposal`算法需要具备另一个特点：**高召回率**（`high recall`），即使生成了大量噪声框，只要最终包含了正确的候选框，那么多一点的噪声也是值得的；

## SS怎么生成候选集的呢？
一般而言，`Region Proposal`算法基于图像分割来生成候选集。在图像分割过程中，将基于一定特征，比如颜色、纹理特征相似的相邻区域进行聚合。明显的用这种从小到大层次化聚类思想实现的算法会比暴力穷举滑动窗口生成的候选集要少很多；`SS`就是这种专门设计用来快速生成候选集的算法，它基于图像的颜色、纹理、以及区域尺寸、形状，从下而上层次化的进行聚合来生成候选集；算法流程如下：
``` python
## Algorithm 1: Hierarchical Grouping Algorithm
Input: (colour) image
Output: Set of object location hypotheses L
Obtain initial regions R = {r1,··· ,rn}  #using [segmentation method](http://cs.brown.edu/~pff/segment/)
Initialise similarity set S to empty
foreach Neighbouring region pair (ri,rj) do
    Calculate similarity s(ri,rj)
    S = S∪s(ri,rj)
while S is not empty do
    Get highest similarity s(ri,rj) = max(S)
    Merge corresponding regions rt = ri ∪rj
    Remove similarities regarding ri: S = S \ s(ri,r∗)
    Remove similarities regarding rj: S = S \ s(r∗,rj)
    Calculate similarity set St between rt and its neighbours
    S = S∪St
    R = R∪rt
Extract object location boxes L from all regions in R
```
可以看到，该算法是由底至上，不停的融合相似度最大的相邻区域，直至最终全部融合为一个区域；观察上述算法，可以看到，算法的关键部分在于如何评价区域的相似度；该算法在计算区域相似度的时候，综合考虑了颜色、纹理、尺寸和形状等，详细可见[Selective Search for Object Recognition](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf)；

一个结果例子：

![object.png](http://pfp.ps.netease.com/kmspvt/file/5cef4af32dcadefa95271e55x2BJ0qtq01?sign=8r_GVHfpKzuO3HOgoc-HCmZM1B0=&expire=1559214584)

## “奇怪”的想法
`SS`算法是为目标检测任务服务而被提出，而根据其算法特性，由`SS`算法生成的候选框有**较大的概率包含物体**，那么当我们把所有候选框的范围累积后，不就可以有较大的概率得到最终的图像的显著性了吗？借助`opencv`，我们尝试做如下实验 [[主体代码来自]](https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/)：
``` python
"""
Usage:
    ./ssearch.py input_image (f|q)
    f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
"""

import sys
import numpy as np
import cv2

if __name__ == '__main__':
    # If image path and f/q is not passed as command
    # line arguments, quit and display help message
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # read image
    im = cv2.imread(sys.argv[1])
    # resize image
    newHeight = 400
    newWidth = int(im.shape[1] * newHeight / im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    # Switch to fast but low recall Selective Search method
    if sys.argv[2] == 'f':
        ss.switchToSelectiveSearchFast()

    # Switch to high recall but slow Selective Search method
    elif sys.argv[2] == 'q':
        ss.switchToSelectiveSearchQuality()
    # if argument is neither f nor q print help message
    else:
        print(__doc__)
        sys.exit(1)

    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    # number of region proposals to show
    numShowRects = 10
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 5

    while True:
        # create a copy of original image
        imOut = im.copy()

        # use ss to detect object by probability of proposed rect
        newImage = np.ones((newHeight, newWidth, 3))

        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if i < numShowRects:
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

                newImage[y:y + h, x:x + w, :] *= 1.3
            else:
                break

        # show output
        cv2.imshow("Output", imOut)
        newImage = np.where(newImage > 255, 255, newImage)
        newImage = cv2.cvtColor(newImage.astype('uint8'), cv2.COLOR_RGB2GRAY)
        cv2.imshow("object", newImage)

        # record key press
        k = cv2.waitKey(0) & 0xFF

        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()
```
实验结果：

![Output.png](http://pfp.ps.netease.com/kmspvt/file/5cef4f6f68d864761f483762SRMMrx3c01?sign=POwfpujSUh_ibE7ZynS6R43uwGc=&expire=1559214584)

![object.png](http://pfp.ps.netease.com/kmspvt/file/5cef4f762dcadedfcedb53fcACNFitFB01?sign=YUAajP4Pb0_4vzmzhm7G6udUxLM=&expire=1559214584)

可以看到，在测试图片上，使用`SS`进行图片显著性检测的结果还可以。

## 所以“卵用”呢？
那么有了上面这种奇怪的使用姿势，我们能用这种姿势干嘛呢？针对电商平台，一个可能的应用就是，我们能够在给定任意图片的情况下，简单快速的检测出图片的主体物体位置，能够**自动的进行裁剪**，用来生成氛围图的banner，在有了图片的显著性位置以后，我们可以根据显著性来放置文案的位置，避免文案对物体的遮挡；

![91113911e74c4355a07bd897a59a6d68.jpg](http://pfp.ps.netease.com/kmspvt/file/5cef51de68d864787daaf515Wfs3EUFk01?sign=W8vr8pAHOdFbIN5oDzqRUjBVwbc=&expire=1559214584) 

图片来自[鹿班](https://luban.aliyun.com)氛围图自动合成banner


#### NOTE: 其他的Region Propocal算法
* [Objectness](http://groups.inf.ed.ac.uk/calvin/objectness/)
* [Constrained Parametric Min-Cuts for Automatic Object Segmentation](http://www.maths.lth.se/matematiklth/personal/sminchis/code/cpmc/index.html)
* [Category Independent Object Proposals](http://vision.cs.uiuc.edu/proposals/)
* [Randomized Prim](http://www.vision.ee.ethz.ch/~smanenfr/rp/index.html)

#### Ref：

* [Selective Search for Object Detection](https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/)
