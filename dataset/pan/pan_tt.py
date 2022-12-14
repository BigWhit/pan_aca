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
from mmcv.parallel import DataContainer as DC

tt_root_dir = './data/totaltext/'
tt_train_data_dir = tt_root_dir + 'Images/Train/'
tt_train_gt_dir = tt_root_dir + 'Groundtruth/Polygon/Train/'
tt_test_data_dir = tt_root_dir + 'Images/Test/'
tt_test_gt_dir = tt_root_dir + 'Groundtruth/Polygon/Test/'


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


def read_mat_lindes(path):
    f = scio.loadmat(path)
    return f


def get_ann(img, gt_path):
    h, w = img.shape[0:2]
    # print(gt_path)
    bboxes = []
    words = []
    # max_distances1 = []  # 最长文本尺度
    max_distances=[]#最长尺度/图片长宽
    sum = 0
    data = read_mat_lindes(gt_path)
    data_polygt = data['polygt']
    l = len(data_polygt)
    # print(l)
    for i, lines in enumerate(data_polygt):
        X = np.array(lines[1])
        Y = np.array(lines[3])

        point_num = len(X[0])
        word = lines[4]
        if len(word) == 0:
            word = '???'
        else:
            word = word[0]
            # word = word[0].encode("utf-8")

        if word == '#':
            word = '###'

        words.append(word)

        arr = np.concatenate([X, Y]).T
        bbox = []
        for i in range(point_num):
            bbox.append(arr[i][0])
            bbox.append(arr[i][1])
        # print(bbox)
        if len(bbox) == 6:
            distance = max(math.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2),
                           math.sqrt((bbox[4] - bbox[0]) ** 2 + (bbox[5] - bbox[1]) ** 2),
                           math.sqrt((bbox[4] - bbox[2]) ** 2 + (bbox[5] - bbox[3]) ** 2)) / (max(h, w))
        elif len(bbox) == 8 or len(bbox) == 10:
            # distance=(math.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) + math.sqrt(
            #     (bbox[6] - bbox[4]) ** 2 + (bbox[7] - bbox[5]) ** 2)) / (3 * max(h, w))
            distance = math.sqrt((bbox[4] - bbox[0]) ** 2 + (bbox[5] - bbox[1]) ** 2) /(max(h, w))
        elif len(bbox) == 12:
            # distance=(math.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) + math.sqrt(
            #     (bbox[4] - bbox[2]) ** 2 + (bbox[5] - bbox[3]) ** 2)+ math.sqrt(
            #     (bbox[8] - bbox[6]) ** 2 + (bbox[9] - bbox[7]) ** 2)+ math.sqrt(
            #     (bbox[10] - bbox[8]) ** 2 + (bbox[11] - bbox[9]) ** 2)) / (3 * max(h, w))
            distance = math.sqrt((bbox[6] - bbox[0]) ** 2 + (bbox[7] - bbox[1]) ** 2) / (max(h, w))
        elif len(bbox) == 14 or len(bbox) == 16 or len(bbox) == 18:
            # distance=(math.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) + math.sqrt(
            #     (bbox[4] - bbox[2]) ** 2 + (bbox[5] - bbox[3]) ** 2)+ math.sqrt(
            #     (bbox[6] - bbox[4]) ** 2 + (bbox[7] - bbox[5]) ** 2) + math.sqrt(
            #     (bbox[10] - bbox[8]) ** 2 + (bbox[11] - bbox[9]) ** 2) + math.sqrt(
            #     (bbox[12] - bbox[10]) ** 2 + (bbox[13] - bbox[11]) ** 2)+math.sqrt(
            #     (bbox[14] - bbox[12]) ** 2 + (bbox[15] - bbox[13]) ** 2)) / (3 * max(h, w))
            distance = math.sqrt((bbox[8] - bbox[0]) ** 2 + (bbox[9] - bbox[1]) ** 2) / (max(h, w))
        elif len(bbox) == 20 or len(bbox) == 22:
            # distance = (math.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) + math.sqrt(
            #     (bbox[4] - bbox[2]) ** 2 + (bbox[5] - bbox[3]) ** 2) + math.sqrt(
            #     (bbox[6] - bbox[4]) ** 2 + (bbox[7] - bbox[5]) ** 2) + math.sqrt(
            #     (bbox[8] - bbox[6]) ** 2 + (bbox[9] - bbox[7]) ** 2) + math.sqrt(
            #     (bbox[12] - bbox[10]) ** 2 + (bbox[13] - bbox[11]) ** 2) + math.sqrt(
            #     (bbox[14] - bbox[12]) ** 2 + (bbox[15] - bbox[13]) ** 2)+ math.sqrt(
            #     (bbox[16] - bbox[14]) ** 2 + (bbox[17] - bbox[15]) ** 2)+ math.sqrt(
            #     (bbox[18] - bbox[16]) ** 2 + (bbox[19] - bbox[17]) ** 2)) / (3 * max(h, w))
            distance = math.sqrt((bbox[10] - bbox[0]) ** 2 + (bbox[11] - bbox[1]) ** 2) / (max(h, w))
        elif len(bbox) == 24 or len(bbox) == 26:
            distance = math.sqrt((bbox[12] - bbox[0]) ** 2 + (bbox[13] - bbox[1]) ** 2) / (max(h, w))
        elif len(bbox) == 28 or len(bbox) == 30:
            distance = math.sqrt((bbox[14] - bbox[0]) ** 2 + (bbox[15] - bbox[1]) ** 2) /(max(h, w))
        else:
            distance = 50 / (max(h, w))
            # print(gt_path)
            # print(bbox)
        sum = sum + distance
        max_distances.append(distance)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * point_num)
        bboxes.append(bbox)
    sum = sum / l  # 平均文本尺度
    # print(sum)
    # print(max_distances)
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
    h, w = img.shape[0:2]

    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
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


def update_word_mask(instance, instance_before_crop, word_mask):
    labels = np.unique(instance)

    for label in labels:
        if label == 0:
            continue
        ind = instance == label
        if np.sum(ind) == 0:
            word_mask[label] = 0
            continue
        ind_before_crop = instance_before_crop == label
        # print(np.sum(ind), np.sum(ind_before_crop))
        if float(np.sum(ind)) / np.sum(ind_before_crop) > 0.9:
            continue
        word_mask[label] = 0

    return word_mask


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


def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char


class PAN_TT(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=640,
                 kernel_scale=0.7,
                 with_rec=False,
                 read_type='pil',
                 report_speed=False):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.with_rec = with_rec
        self.read_type = read_type

        if split == 'train':
            data_dirs = [tt_train_data_dir]
            gt_dirs = [tt_train_gt_dir]
        elif split == 'test':
            data_dirs = [tt_test_data_dir]
            gt_dirs = [tt_test_gt_dir]
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

                gt_name = 'poly_gt_' + img_name.split('.')[0] + '.mat'
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

        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE')
        self.max_word_num = 200
        self.max_word_len = 32

    def __len__(self):
        return len(self.img_paths)

    def get_full_lexicon(self):
        full_lexicon = set()
        for gt_path in self.gt_paths:
            data = read_mat_lindes(gt_path)
            data_polygt = data['polygt']
            for i, lines in enumerate(data_polygt):
                word = lines[4]
                if len(word) == 0:
                    continue

                word = word[0]
                if word == '#':
                    continue

                full_lexicon.add(word.lower())

        return sorted(list(full_lexicon))

    def prepare_train_data(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]
        # print(gt_path)
        img = get_img(img_path, self.read_type)
        bboxes, words, max_distances = get_ann(img, gt_path)
        # print(max_distances)
        ###计算平均文本实例size/单个文本实例size
        # max_distances = []  # 平均文本尺度/单个文本尺度
        # for index in max_distances1:
        #     mean = index/sum  # (0.3-3.8)
        #     max_distances.append(mean)
        # print(max_distances)
        t = len(max_distances)
        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]

        gt_words = np.full((self.max_word_num + 1, self.max_word_len), self.char2id['PAD'], dtype=np.int32)
        word_mask = np.zeros((self.max_word_num + 1,), dtype=np.int32)
        for i, word in enumerate(words):
            if word == '###' or word == '???':
                continue
            word = word.lower()
            gt_word = np.full((self.max_word_len,), self.char2id['PAD'], dtype=np.int)
            for j, char in enumerate(word):
                if j > self.max_word_len - 1:
                    break
                if char in self.char2id:
                    gt_word[j] = self.char2id[char]
                else:
                    gt_word[j] = self.char2id['UNK']
            if len(word) > self.max_word_len - 1:
                gt_word[-1] = self.char2id['EOS']
            else:
                gt_word[len(word)] = self.char2id['EOS']
            gt_words[i + 1] = gt_word
            word_mask[i + 1] = 1

        if self.is_transform:
            img = random_scale(img, self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        miu = np.zeros(img.shape[0:2], dtype='float32')
        if len(bboxes) > 0:
            for i in range(len(bboxes)):
                bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] // 2)),
                                       (bboxes[i].shape[0] // 2, 2)).astype('int32')  # 转换成坐标形式
                # miu_i[i]=np.reshape(bboxes[i]*)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
            for i in range(len(bboxes)):
                cv2.drawContours(miu, [bboxes[i]], -1, max_distances[i], -1)

        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask, miu]  # 加入miu
            imgs.extend(gt_kernels)

            if not self.with_rec:
                imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            gt_instance_before_crop = imgs[1].copy()
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, miu, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3], imgs[4:]  # 加入miu
            word_mask = update_word_mask(gt_instance, gt_instance_before_crop, word_mask)

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        ###lossv1方法
        b = [0 for i in range(self.max_word_num - t)]
        a=[0]
        # b_t = torch.tensor(b, dtype=torch.float32)
        max_distances = a+max_distances + b
        max_distances = np.asarray(max_distances, dtype=np.float32)

        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()
        gt_words = torch.from_numpy(gt_words).long()
        word_mask = torch.from_numpy(word_mask).long()
        # miu = torch.from_numpy(miu).float()
        max_distances = torch.from_numpy(max_distances).float()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
            max_distances=max_distances,
            # miu=miu
        )
        if self.with_rec:
            data.update(dict(
                gt_words=gt_words,
                word_masks=word_mask
            ))

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
