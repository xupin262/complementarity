# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import cv2
import mmcv
import numpy as np
import torch
from PIL import Image
import math
import pdb
import shutil
import os

def RemoveDir(filepath):
    # 如果文件存在就清空,如果文件夹不存在就创建.
    if os.path.exists(filepath):
        shutil.rmtree(filepath)
    os.mkdir(filepath)

@torch.no_grad()
def main(args):
    loaded_dicts = []
    # 获取预测结果的pt文件
    predict_pts = args.predict_pt

    # # # 直接给出计算好的权重
    weight = args.dices
    print("weight: ", weight)

    for predict_pt in predict_pts:
        loaded_dict = torch.load(predict_pt)
        loaded_dicts.append(loaded_dict)
        torch.cuda.empty_cache()
        tmpdir = args.out
        # 如果存在，就删除；如果不存在，就创建该目录
        RemoveDir(tmpdir)
    img_names = []
    for key in loaded_dicts[0]:
        img_names.append(key)

    for i in img_names: # 遍历图片
        result_logits = 0
        for j, loaded_dict in enumerate(loaded_dicts):
            # pdb.set_trace()
            x = torch.softmax(loaded_dict[str(i)], dim=0).cpu().numpy()
            # x = loaded_dict[str(i)].cpu().numpy()
            result_logits += weight[j]*x
        pred = result_logits.argmax(axis=0) # np(480, 640) 0or1
        file_name = os.path.join(tmpdir, str(i)+'.png')
        # pdb.set_trace()
        Image.fromarray(pred.astype(np.uint8)*255).save(file_name)
    print('done!')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model Ensemble with logits result')
    parser.add_argument(
        '--predict_pt',
        type=str,
        nargs='+',
        help='ensemble predict_pt files path')
    parser.add_argument(
        '--out', type=str, default='results', help='the dir to save result')
    parser.add_argument(
        '--dices', type=float, nargs='+',  help='dices of models to use') 
    args = parser.parse_args()
    assert len(args.predict_pt) == len(args.dices), \
        f'len(predict_pt) must equal len(dices), ' \
        f'but len(predict_pt) = {len(args.predict_pt)} and'\
        f'len(dices) = {len(args.dices)}'
    assert args.out, "ensemble result out-dir can't be None"
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)


# 用法示例如下：
# CUDA_VISIBLE_DEVICES=0 python tools/model_ensemble_HB.py \
#   --predict_pt 'work_dirs/pt/mask2former_733_predict.pt'\
#                 'work_dirs/pt/upernet_7191_predict.pt'\
#   --out 'work_dirs/ensemble/ensemble_HB_mask2former-733_upernet-7191' \
#   --dices    0.733 0.267





