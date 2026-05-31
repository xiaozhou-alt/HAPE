from __future__ import division, print_function, absolute_import
import glob
import warnings
import os.path as osp
import os
from .bases import BaseImageDataset


class Market1501MM(BaseImageDataset):
    """
    Market1501 多模态版本（RGB, NIR, TIR）
    数据集结构：
        root/
            train/
                RGB/
                NI/
                TI/
            query/
                RGB/
                NI/
                TI/
            gallery/
                RGB/
                NI/
                TI/
    文件命名格式：0002_c1s1_000451_03.jpg
    其中：前4位为pid，c1表示摄像头1（camid=1）
    """
    dataset_dir = 'Market1501MM'   # 总文件夹名称

    def __init__(self, root='', verbose=True, **kwargs):
        super(Market1501MM, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501MM loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        for split in ['train', 'query', 'gallery']:
            split_dir = osp.join(self.dataset_dir, split)
            if not osp.exists(split_dir):
                raise RuntimeError(f"'{split_dir}' is not available")
            for modality in ['RGB', 'NI', 'TI']:
                mod_dir = osp.join(split_dir, modality)
                if not osp.exists(mod_dir):
                    raise RuntimeError(f"'{mod_dir}' is not available")

    def _process_dir(self, dir_path, relabel=False):
        # 获取所有RGB图片路径
        img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
        pid_container = set()
        for img_path_RGB in img_paths_RGB:
            fname = os.path.basename(img_path_RGB)
            # 文件名格式：0002_c1s1_000451_03.jpg
            pid = int(fname.split('_')[0])   # 前4位数字
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path_RGB in img_paths_RGB:
            fname = os.path.basename(img_path_RGB)
            # 构造NI和TI路径
            img_path_NI = osp.join(dir_path, 'NI', fname)
            img_path_TI = osp.join(dir_path, 'TI', fname)

            # 解析pid和camid
            pid = int(fname.split('_')[0])
            cam_part = fname.split('_')[1]   # 例如 'c1s1'
            camid = int(cam_part[1]) - 1      # 从0开始计数

            trackid = -1   # 无序列信息

            if relabel:
                pid = pid2label[pid]

            data.append(([img_path_RGB, img_path_NI, img_path_TI], pid, camid, trackid))

        return data