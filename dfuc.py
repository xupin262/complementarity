# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import pdb


@DATASETS.register_module()
class DFUCDataset(BaseSegDataset):
    """DFUC dataset.
    """
    METAINFO = dict(
        classes=('background', 'dfu'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                #  reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            # reduce_zero_label=reduce_zero_label,
            **kwargs)
        # pdb.set_trace()
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)


# (Pdb) self.data_prefix['img_path']
# '/data/xupin/work/mmsegmentation/dataset/DFUC20022byVOC/JPEGImages'
# (Pdb) self.backend_args
# (Pdb) self.ann_file
# '/data/xupin/work/mmsegmentation/dataset/DFUC20022byVOC/ImageSets/Segmentation/train.txt'
# (Pdb) self.backend_args
# (Pdb) fileio.exists(self.data_prefix['img_path'],
# *** SyntaxError: unexpected EOF while parsing
# (Pdb) fileio.exists(self.data_prefix['img_path'], self.backend_args)
# True
# (Pdb) osp.isfile(self.ann_file)
# True