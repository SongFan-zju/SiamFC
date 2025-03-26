import numpy as np
import cv2
import torch
import numbers


def crop_and_resize(
        img,
        center,
        size,
        out_size,
        border_type=cv2.BORDER_CONSTANT,
        border_value=(0, 0, 0),  #border_value使用的是图像均值(averageR,aveG,aveB)
        interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)  #对size取整
    corners = np.concatenate((  #np.concatenate:数组的凭借 np.concatenate((a,b),axis)  axis=0是列拼接，axis=1是行拼接 省略axis为0
        np.round(center - (size - 1) / 2), np.round(center - (size - 1) / 2) + size))  #得到corners=[ymin,xmin,ymax,xmax]
    corners = np.round(corners).astype(int)  #转化为int型
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('original', img)
    # cv2.imwrite('original.png', img)
    #填充
    pads = np.concatenate((-corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))  #得到4个值中最大的与0对比
    if npad > 0:
        img = cv2.copyMakeBorder(img, npad, npad, npad, npad, border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)  #如果经行了填充，那么中心坐标也要变
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]  #得到patch的大小
    patch = cv2.resize(patch, (out_size, out_size), interpolation=interp)
    return patch


class Compose(object):  # 继承了object类，就拥有了object类里面好多可以操作的对象

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):  # 为了将类的实例对象变为可调用对象（相当于重载()运算符）  a=Compose() a.__call__()   和a()的使用是一样的
        for t in self.transforms:
            img = t(img)
        return img


# 主要是随机的resize图片的大小，变化再[1 1.05之内]其中要注意cv2.resize()的一点用法
class RandomStretch(object):

    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch

    def __call__(self, img):
        interp = np.random.choice([  # 调用interp时候随机选择一个
            cv2.INTER_LINEAR,  # 双线性插值（默认设置）
            cv2.INTER_CUBIC,  # 4x4像素领域的双三次插值
            cv2.INTER_AREA,  # 像素区域关系重采样，类似与NEAREST
            cv2.INTER_NEAREST,  # 最近领插值
            cv2.INTER_LANCZOS4
        ])  # 8x8像素的Lanczosc插值
        scale = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        out_size = (
            round(img.shape[1] * scale),  # 这里是width
            round(img.shape[0] * scale))  # 这里是heigth  cv2的用法导致
        return cv2.resize(img, out_size, interpolation=interp)  # 将img的大小resize成out_size


# 从img中心抠一块(size, size)大小的patch，如果不够大，以图片均值进行pad之后再crop
class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):  # isinstance(object, classinfo) 判断实例是否是这个类或者object是变量
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]  # img.shape为[height,width,channel]
        tw, th = self.size
        i = round((h - th) / 2.)  # round(x,n) 对x四舍五入，保留n位小数 省略n 0位小数
        j = round((w - tw) / 2.)

        npad = max(0, -i, -j)
        if npad > 0:
            avg_color = np.mean(img, axis=(0, 1))  # 取整个图片的像素均值
            img = cv2.copyMakeBorder(  # 添加边框函数，上下左右要扩展的像素数都是npad,BORDER_CONSTANT固定值填充，值为avg_color）
                img,
                npad,
                npad,
                npad,
                npad,
                cv2.BORDER_CONSTANT,
                value=avg_color)
            i += npad
            j += npad
        return img[i:i + th, j:j + tw]


# 用法类似CenterCrop，只不过从随机的位置抠，没有pad的考虑
class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return img[i:i + th, j:j + tw]


# 就是字面意思，把np.ndarray转化成torch tensor类型
class ToTensor(object):

    def __call__(self, img):
        return torch.from_numpy(img).float().permute((2, 0, 1))


class Transformer(object):

    def __init__(self, exemplar_sz=127, instance_sz=255, context=0.5):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context
        self.transforms_z = Compose([
            RandomStretch(),  # 随机resize图片大小,变化再[1 1.05]之内
            CenterCrop(instance_sz - 8),  # 中心裁剪 裁剪为255-8
            RandomCrop(instance_sz - 2 * 8),  # 随机裁剪  255-8->255-8-8
            CenterCrop(exemplar_sz),  # 中心裁剪 255-8-8->127
            ToTensor()
        ])  # 图片的数据格式从numpy转换成torch张量形式
        self.transforms_x = Compose([
            RandomStretch(),  # s随机resize图片
            CenterCrop(instance_sz - 8),  # 中心裁剪 裁剪为255-8
            RandomCrop(instance_sz - 2 * 8),  # 随机裁剪 255-8->255-8-8
            ToTensor()
        ])  # 图片数据格式转化为torch张量

    def _crop(self, img, box, out_size):
        box = np.array(
            [
                box[1] - 1 + (box[3] - 1) / 2,  # y_center
                box[0] - 1 + (box[2] - 1) / 2,  # x_center
                box[3],  # height
                box[2]  # width
            ],
            dtype=np.float32)
        center, target_sz = box[:2], box[2:]
        context = self.context * np.sum(target_sz)  # context = 0.5*(w + h)
        size = np.sqrt(np.prod(target_sz + context))  # size = sqrt((w+c)*(h+c))
        size *= out_size / self.exemplar_sz  # 尺寸归一化到 output 尺寸比例
        avg_color = np.mean(img, axis=(0, 1), dtype=float)
        interp = np.random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4])
        patch = crop_and_resize(img, center, size, out_size, border_value=avg_color, interp=interp)
        return patch

    def __call__(self, z, x, bbox_z, bbox_x):
        z = self._crop(z, bbox_z, self.instance_sz)
        x = self._crop(x, bbox_x, self.instance_sz)
        z = self.transforms_z(z)
        x = self.transforms_x(x)
        return z, x
