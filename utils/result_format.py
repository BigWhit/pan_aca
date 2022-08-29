import os
import os.path as osp
import zipfile
import cv2
import numpy as np


class ResultFormat(object):
    def __init__(self, data_type, result_path):
        self.data_type = data_type
        self.result_path = result_path

        if osp.isfile(result_path):
            os.remove(result_path)

        if result_path.endswith('.zip'):
            result_path = result_path.replace('.zip', '')

        if not osp.exists(result_path):
            os.makedirs(result_path)

    def write_result(self, img_name, img, outputs):  ######修改,添加img
        if 'IC15' in self.data_type:
            self._write_result_ic15(img_name, img, outputs)
        elif 'TT' in self.data_type:
            self._write_result_tt(img_name, img, outputs)
        elif 'CTW' in self.data_type:
            self._write_result_ctw(img_name, img, outputs)  ######修改,添加img
        elif 'MSRA' in self.data_type:
            self._write_result_msra(img_name,img, outputs)
        elif 'ART' in self.data_type:
            self._write_result_art(img_name, img, outputs)
        elif 'COCO' in self.data_type:
            self._write_result_coco(img_name, img, outputs)
        elif 'LSVT' in self.data_type:
            self._write_result_lsvt(img_name, img, outputs)

    def _write_result_ic15(self, img_name, img, outputs):
        assert self.result_path.endswith('.zip'), 'Error: ic15 result should be a zip file!'
        # 加
        save_img_folder = osp.join(self.result_path.replace('.zip', ''), 'img')
        if not osp.exists(save_img_folder):
            os.makedirs(save_img_folder)
        save_txt_folder = osp.join(self.result_path.replace('.zip', ''), 'txt')
        if not osp.exists(save_txt_folder):
            os.makedirs(save_txt_folder)
            # 加
        # tmp_folder = self.result_path.replace('.zip', '')

        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = "%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(values)
            lines.append(line)

        file_name = 'res_%s.txt' % img_name
        # file_path = osp.join(tmp_folder, file_name)
        file_path = osp.join(save_txt_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

        z = zipfile.ZipFile(self.result_path, 'a', zipfile.ZIP_DEFLATED)
        z.write(file_path, file_name)
        z.close()

        # 增加图片img
        img = np.array(img)
        file_img_name = 'res_%s.jpg' % img_name
        file_img_path = osp.join(save_img_folder, file_img_name)
        with open(file_path, 'r') as f:
            for i, bbox in enumerate(bboxes):
                # print(bbox)
                poly = bbox.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

        cv2.imwrite(file_img_path, img)

    ############

    def _write_result_tt(self, image_name, img, outputs):
        # 加
        save_img_folder = osp.join(self.result_path.replace('.zip', ''), 'img')
        if not osp.exists(save_img_folder):
            os.makedirs(save_img_folder)
        save_txt_folder = osp.join(self.result_path.replace('.zip', ''), 'txt')
        if not osp.exists(save_txt_folder):
            os.makedirs(save_txt_folder)
            # 加
        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ",%d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(save_txt_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)
        #######增加图片输出
        img = np.array(img).copy()
        file_img_name = 'res_%s.jpg' % image_name
        file_img_path = osp.join(save_img_folder, file_img_name)
        with open(file_path, 'r') as f:
            for i, bbox in enumerate(bboxes):
                poly = bbox.reshape(-1, 2)
                # print(bbox)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

        cv2.imwrite(file_img_path, img)

    ############

    def _write_result_ctw(self, image_name, img, outputs):  ######修改,添加img
        # 加
        save_img_folder = osp.join(self.result_path.replace('.zip', ''), 'img')
        if not osp.exists(save_img_folder):
            os.makedirs(save_img_folder)
        save_txt_folder = osp.join(self.result_path.replace('.zip', ''), 'txt')
        if not osp.exists(save_txt_folder):
            os.makedirs(save_txt_folder)
            # 加

        bboxes = outputs['bboxes']
        # print(bboxes)
        lines = []
        for i, bbox in enumerate(bboxes):
            # print(bbox)
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ",%d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(save_txt_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

        #######增加图片输出
        img = np.array(img)
        file_img_name = 'res_%s.jpg' % image_name
        file_img_path = osp.join(save_img_folder, file_img_name)
        with open(file_path, 'r') as f:
            for i, bbox in enumerate(bboxes):
                poly = bbox.reshape(-1, 2)
                # print(bbox)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

        cv2.imwrite(file_img_path, img)

    ############

    def _write_result_coco(self, image_name, img, outputs):
        # 加
        save_img_folder = osp.join(self.result_path.replace('.zip', ''), 'img')
        if not osp.exists(save_img_folder):
            os.makedirs(save_img_folder)
        save_txt_folder = osp.join(self.result_path.replace('.zip', ''), 'txt')
        if not osp.exists(save_txt_folder):
            os.makedirs(save_txt_folder)
            # 加

        bboxes = outputs['bboxes']
        # print(bboxes)
        lines = []
        for i, bbox in enumerate(bboxes):
            # print(bbox)
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ",%d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(save_txt_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

        #######增加图片输出
        img = np.array(img).copy()
        file_img_name = 'res_%s.jpg' % image_name
        file_img_path = osp.join(save_img_folder, file_img_name)
        with open(file_path, 'r') as f:
            for i, bbox in enumerate(bboxes):
                poly = bbox.reshape(-1, 2)
                # print(bbox)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

        cv2.imwrite(file_img_path, img)

    ############

    def _write_result_msra(self, image_name,img, outputs):
        # 加
        save_img_folder = osp.join(self.result_path.replace('.zip', ''), 'img')
        if not osp.exists(save_img_folder):
            os.makedirs(save_img_folder)
        save_txt_folder = osp.join(self.result_path.replace('.zip', ''), 'txt')
        if not osp.exists(save_txt_folder):
            os.makedirs(save_txt_folder)
            # 加
        bboxes = outputs['bboxes']

        lines = []
        for b_idx, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ", %d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(save_txt_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)
 #######增加图片输出
        img = np.array(img).copy()
        file_img_name = 'res_%s.jpg' % image_name
        file_img_path = osp.join(save_img_folder, file_img_name)
        with open(file_path, 'r') as f:
            for i, bbox in enumerate(bboxes):
                poly = bbox.reshape(-1, 2)
                # print(bbox)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=4)

        cv2.imwrite(file_img_path, img)

    ############

    def _write_result_art(self, img_name, img, outputs):
        save_img_folder = osp.join(self.result_path.replace('.zip', ''), 'img')
        if not osp.exists(save_img_folder):
            os.makedirs(save_img_folder)
        save_txt_folder = osp.join(self.result_path.replace('.zip', ''), 'txt')
        if not osp.exists(save_txt_folder):
            os.makedirs(save_txt_folder)

        # 文本输出
        bboxes = outputs['bboxes']
        # print(bboxes)
        lines = []
        for i, bbox in enumerate(bboxes):
            # print(bbox)
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ",%d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % img_name
        file_path = osp.join(save_txt_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

        #####图片输出
        img_ = np.array(img).copy()
        file_img_name = 'res_%s.jpg' % img_name
        file_img_path = osp.join(save_img_folder, file_img_name)
        with open(file_path, 'r') as f:
            for i, bbox in enumerate(bboxes):
                poly = bbox.reshape(-1, 2)
                # print(bbox)
                cv2.polylines(img_, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

        cv2.imwrite(file_img_path, img_)

    def _write_result_lsvt(self, img_name, img, outputs):
        save_img_folder = osp.join(self.result_path.replace('.zip', ''), 'img')
        if not osp.exists(save_img_folder):
            os.makedirs(save_img_folder)
        save_txt_folder = osp.join(self.result_path.replace('.zip', ''), 'txt')
        if not osp.exists(save_txt_folder):
            os.makedirs(save_txt_folder)

        # 文本输出
        bboxes = outputs['bboxes']
        # print(bboxes)
        lines = []
        for i, bbox in enumerate(bboxes):
            # print(bbox)
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ",%d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % img_name
        file_path = osp.join(save_txt_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

        #####图片输出
        img_ = np.array(img).copy()
        file_img_name = 'res_%s.jpg' % img_name
        file_img_path = osp.join(save_img_folder, file_img_name)
        with open(file_path, 'r') as f:
            for i, bbox in enumerate(bboxes):
                poly = bbox.reshape(-1, 2)
                # print(bbox)
                cv2.polylines(img_, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

        cv2.imwrite(file_img_path, img_)
