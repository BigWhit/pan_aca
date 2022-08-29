import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import math
import string
import scipy.io as scio
import mmcv
import os

ctw_root_dir = './data/CTW1500/'
# ctw_train_data_dir = ctw_root_dir + 'train/text_image/'
ctw_train_data_dir = './data/longtxt/train/image/'
ctw_train_gt_dir = ctw_root_dir + 'train/text_label_curve/'
# ctw_test_data_dir = ctw_root_dir + 'test/text_image/'
ctw_test_data_dir = './data/longtxt/test/image/'  # 选择test部分图片
ctw_test_gt_dir = ctw_root_dir + 'test/text_label_circum/'


def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception as e:
        print(img_path)
        raise
    return img


def get_ann(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    # print(gt_path)
    l = len(lines)
    bboxes = []
    words = []
    max_distances = []
    left = []
    min_distances = []
    sum = 0
    for line in lines:
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])
        left.append(x1)
        left.append(y1)
        # lossv1
        x2 = np.int(gt[2])
        y2 = np.int(gt[3])
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / (3 * max(h, w))
        ###
        # print(distance)
        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)  # 28个值
        # print(bbox)
        ##lossv2
        # distance = (math.sqrt((bbox[2]-bbox[0])**2+(bbox[3]-bbox[1])**2) + math.sqrt((bbox[4]-bbox[2])**2+(bbox[5]-bbox[3])**2)+
        #            math.sqrt((bbox[6]-bbox[4])**2+(bbox[7]-bbox[5])**2) + math.sqrt((bbox[8]-bbox[6])**2+(bbox[9]-bbox[7])**2)+
        #            math.sqrt((bbox[10]-bbox[8])**2+(bbox[11]-bbox[9])**2) + math.sqrt((bbox[12]-bbox[10])**2+(bbox[13]-bbox[11])**2)+
        #            math.sqrt((bbox[16]-bbox[14])**2+(bbox[17]-bbox[15])**2) + math.sqrt((bbox[18]-bbox[16])**2+(bbox[19]-bbox[17])**2)+
        #            math.sqrt((bbox[20]-bbox[18])**2+(bbox[21]-bbox[19])**2) + math.sqrt((bbox[22]-bbox[20])**2+(bbox[23]-bbox[21])**2)+
        #            math.sqrt((bbox[24]-bbox[22])**2+(bbox[25]-bbox[23])**2) + math.sqrt((bbox[26]-bbox[24])**2+(bbox[27]-bbox[25])**2))/(4 * max(h, w))
        ###
        ####lossv3
        # distance = (math.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) + math.sqrt(
        #     (bbox[4] - bbox[2]) ** 2 + (bbox[5] - bbox[3]) ** 2) +
        #             math.sqrt((bbox[6] - bbox[4]) ** 2 + (bbox[7] - bbox[5]) ** 2) + math.sqrt(
        #             (bbox[8] - bbox[6]) ** 2 + (bbox[9] - bbox[7]) ** 2) +
        #             math.sqrt((bbox[10] - bbox[8]) ** 2 + (bbox[11] - bbox[9]) ** 2) + math.sqrt(
        #             (bbox[12] - bbox[10]) ** 2 + (bbox[13] - bbox[11]) ** 2) +
        #             math.sqrt((bbox[16] - bbox[14]) ** 2 + (bbox[17] - bbox[15]) ** 2) + math.sqrt(
        #             (bbox[18] - bbox[16]) ** 2 + (bbox[19] - bbox[17]) ** 2) +
        #             math.sqrt((bbox[20] - bbox[18]) ** 2 + (bbox[21] - bbox[19]) ** 2) + math.sqrt(
        #             (bbox[22] - bbox[20]) ** 2 + (bbox[23] - bbox[21]) ** 2) +
        #             math.sqrt((bbox[24] - bbox[22]) ** 2 + (bbox[25] - bbox[23]) ** 2) + math.sqrt(
        #             (bbox[26] - bbox[24]) ** 2 + (bbox[27] - bbox[25]) ** 2)) / 2
        ####lossv4平均宽度
        # distance = (abs(bbox[3] - bbox[25]) + abs(bbox[5] - bbox[23]) + abs(bbox[7] - bbox[21]) + abs(
        #     bbox[9] - bbox[19]) + abs(bbox[11] - bbox[17])) / 5
        # if distance == 0.0:
        #     distance = distance + 1
        # print('dis',distance)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)
        sum = sum + distance
        bboxes.append(bbox)
        words.append('???')
        max_distances.append(distance)

    # print('%s是' % gt_path)
    # print(bboxes)
    # print(left)
    min_distances.append(0.0)
    for i in range(0, len(left) - 3, 2):
        min_distance = math.sqrt((left[i] - left[i + 2]) ** 2 + (left[i + 1] - left[i + 3]) ** 2) / max(h, w)
        min_distances.append(min_distance)
    # print(min_distances)
    # print('max', max_distances)
    sum = sum / l
    # print('mean',sum)
    return bboxes, words, max_distances


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=640):
    h, w = img.shape[0:2]  # 图片原始大小
    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])  # 随机变换尺寸
    scale = (np.random.choice(random_scale) * short_size) / min(h, w)

    img = scale_aligned(img, scale)
    return img


def scale_aligned_short(img, short_size=640):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


class PAN_CTW(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=640,
                 kernel_scale=0.7,
                 read_type='pil',
                 report_speed=False):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.read_type = read_type

        if split == 'train':
            data_dirs = [ctw_train_data_dir]
            gt_dirs = [ctw_train_gt_dir]
        elif split == 'test':
            data_dirs = [ctw_test_data_dir]
            gt_dirs = [ctw_test_gt_dir]
        else:
            print('Error: split must be train or test!')
            raise

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = [img_name for img_name in mmcv.utils.scandir(data_dir, '.jpg')]
            img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.png')])

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

        if report_speed:
            target_size = 3000
            data_size = len(self.img_paths)
            extend_scale = (target_size + data_size - 1) // data_size
            self.img_paths = (self.img_paths * extend_scale)[:target_size]
            self.gt_paths = (self.gt_paths * extend_scale)[:target_size]

        self.max_word_num = 200

    def __len__(self):
        return len(self.img_paths)

    def prepare_train_data(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]
        img = get_img(img_path, self.read_type)
        # print(img.shape[0:3])#不规则大小
        # bboxes, words, max_distances1,sum  = get_ann(img, gt_path)  # 加入max_distance
        bboxes, words, max_distances = get_ann(img, gt_path)

        ###计算平均文本实例size/单个文本实例size
        # max_distances = []
        # for index in max_distances1:
        #     mean = sum/index#(0.3-3.8)
        #     max_distances.append(mean)
        # print(max_distances)
        t = len(max_distances)
        # print(len(bboxes))#有多少文本实例
        # print(bboxes)
        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]

        if self.is_transform:
            img = random_scale(img, self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')  # 随机变换尺寸后的图片大小,全0
        # print(len(gt_instance))  # img.shape[0]
        training_mask = np.ones(img.shape[0:2], dtype='uint8')  # 全1
        miu = np.zeros(img.shape[0:2], dtype='float32')
        if len(bboxes) > 0:  # 有多少文本实例
            for i in range(len(bboxes)):
                bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] // 2)),
                                       (bboxes[i].shape[0] // 2, 2)).astype('int32')  # 14个坐标
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)  # -1绘制所有轮廓，-1为填充模式,gt_instance是 gt的轮廓图,i+100
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
            for i in range(len(bboxes)):
                cv2.drawContours(miu, [bboxes[i]], -1, max_distances[i], -1)

            # #####输出gt的轮廓图
            # save_img_folder = 'C:/Users/Administrator/Desktop/pan_pp.pytorch/ctw/gt_instance'
            # if not os.path.exists(save_img_folder):
            #     os.makedirs(save_img_folder)
            # file_img_name = img_path[-8:]
            # file_img_path = os.path.join(save_img_folder, file_img_name)
            # cv2.imwrite(file_img_path, gt_instance)
            #######

        gt_kernels = []
        for rate in [self.kernel_scale]:  # kernel_scale=0.7,rate=0.7
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)  # i+100
            gt_kernels.append(gt_kernel)

            # #####输出kernel的轮廓图
            # save_img_folder1 = 'C:/Users/Administrator/Desktop/pan_pp.pytorch/ctw/gt_kernel'
            # if not os.path.exists(save_img_folder1):
            #     os.makedirs(save_img_folder1)
            # file_img_name1 = img_path[-8:]
            # file_img_path1 = os.path.join(save_img_folder1, file_img_name1)
            # cv2.imwrite(file_img_path1, gt_kernel)
            #######

        if self.is_transform:
            # print(len(gt_instance))#图片尺寸
            # print(gt_instance)#背景全0，文字坐标同一个文字实例对应一个数字【1, max_instance + 1】
            imgs = [img, gt_instance, training_mask, miu]
            imgs.extend(gt_kernels)

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, miu, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3], imgs[4:]
            # print(len(gt_instance))  # 统一成640
            # print(img.shape[0:3])#（640，640，3）
        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        # print('gt_text is', gt_text)#背景是0，文字是1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        # print('max_instance is ', max_instance)#文本实例个数
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)  # 201*4的全0矩阵
        for i in range(1, max_instance + 1):
            ind = gt_instance == i  # 单个文本实例对应的位置
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))  # 转置  单个文本实例对应的像素位置
            tl = np.min(points, axis=0)  # 最小横坐标
            br = np.max(points, axis=0) + 1  # 最大横坐标
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])  # 左上右下两个点

        b = [0 for i in range(self.max_word_num - t)]
        a = [0]
        max_distances = a + max_distances + b
        # min_distances = a+min_distances + b
        max_distances = np.asarray(max_distances, dtype=np.float32)
        # min_distances = np.asarray(min_distances, dtype=np.float32)

        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        # print(img.size())#(3,640,640)
        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()
        max_distances = torch.from_numpy(max_distances).float()
        # min_distances = torch.from_numpy(min_distances).float()
        # miu = torch.from_numpy(miu).float()
        # print(min_distances)
        # print(gt_bboxes.size())#[201,4]
        # print(max_distances.size())#[201]
        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
            max_distances=max_distances,
            # min_distances=min_distances
            # miu=miu
        )

        return data

    def prepare_test_data(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path, self.read_type)
        img_meta = dict(
            org_img_size=np.array(img.shape[:2])
        )

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(
            img_size=np.array(img.shape[:2])
        ))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        data = dict(
            imgs=img,
            img_metas=img_meta
        )

        return data

    def __getitem__(self, index):
        if self.split == 'train':
            return self.prepare_train_data(index)
        elif self.split == 'test':
            return self.prepare_test_data(index)
