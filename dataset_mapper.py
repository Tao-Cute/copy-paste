import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import pycocotools.mask as mask_util

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data.detection_utils import transform_keypoint_annotations
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.structures import Keypoints, PolygonMasks, BitMasks
from fvcore.transforms.transform import TransformList
from .custom_build_augmentation import build_custom_augmentation
# from .tar_dataset import DiskTarDataset

import random


__all__ = ["CustomDatasetMapper"]

class CustomDatasetMapper(DatasetMapper):
    @configurable
    def __init__(self, is_train: bool, 
        with_ann_type=False,
        dataset_ann=[],
        use_diff_bs_size=False,
        dataset_augs=[],
        is_debug=False,
        use_tar_dataset=False,
        tarfile_path='',
        tar_index_dir='',
        dataset_size = 640,
        **kwargs):
        """
        add image labels
        """
        self.with_ann_type = with_ann_type
        self.dataset_ann = dataset_ann
        self.use_diff_bs_size = use_diff_bs_size
        if self.use_diff_bs_size and is_train:
            self.dataset_augs = [T.AugmentationList(x) for x in dataset_augs]
        self.is_debug = is_debug
        self.use_tar_dataset = use_tar_dataset
        self.dataset_size = dataset_size
        if self.use_tar_dataset:
            print('Using tar dataset')
            self.tar_dataset = DiskTarDataset(tarfile_path, tar_index_dir)
        super().__init__(is_train, **kwargs)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret.update({
            'with_ann_type': cfg.WITH_IMAGE_LABELS,
            'dataset_ann': cfg.DATALOADER.DATASET_ANN,
            'use_diff_bs_size': cfg.DATALOADER.USE_DIFF_BS_SIZE,
            'is_debug': cfg.IS_DEBUG,
            'use_tar_dataset': cfg.DATALOADER.USE_TAR_DATASET,
            'tarfile_path': cfg.DATALOADER.TARFILE_PATH,
            'tar_index_dir': cfg.DATALOADER.TAR_INDEX_DIR,
            'dataset_sizes' : cfg.DATALOADER.DATASET_INPUT_SIZE,
        })
        if ret['use_CopyAndPaste'] and is_train:
            # --------------------------------------- #
            # TODO
            # 加入ImageNet信息
            # --------------------------------------- #
            if cfg.INPUT.CUSTOM_AUG == 'CopyAndPaste21K':
                ret['ImageNet'] = 'copyandpaste'
            else:
                assert cfg.INPUT.CUSTOM_AUG == 'ResizeShortestEdge'
                min_sizes = cfg.DATALOADER.DATASET_MIN_SIZES
                max_sizes = cfg.DATALOADER.DATASET_MAX_SIZES
                ret['dataset_augs'] = [
                    build_custom_augmentation(
                        cfg, True, min_size=mi, max_size=ma) \
                        for mi, ma in zip(min_sizes, max_sizes)]
        else:
            ret['dataset_augs'] = []

        return ret


    def reshape_image(self, ori_input, dataset_dict = None, seg = None, bbox=None):
        resize_scale = random.uniform(0.3, 1.5)
        h, w = ori_input.shape[:2]
        new_h = int(h * resize_scale)
        new_w = int(w * resize_scale)
        
        # ori_image_reshape
        if dataset_dict is not None:
            augs = T.AugmentationList([
            T.RandomFlip(prob=0.5),
            T.ResizeTransform(h, w, new_h, new_w),
            ])
            input = T.AugInput(ori_input, sem_seg=seg)
            transform = augs(input)  # type: T.Transform
            image_transformed = input.image  # new image
            seg_transformed = input.sem_seg

            image_shape = image_transformed.shape[:2]
            all_annos = [
                    (utils.transform_instance_annotations(
                        obj, transform, image_shape, 
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                    ),  obj.get("iscrowd", 0))
                    for obj in dataset_dict.pop("annotations")
                ]
            # new annotations
            annos = [ann[0] for ann in all_annos if ann[1] == 0]

            bbox_transformed = np.asarray([obj['bbox'] for obj in annos])

            ins_cls = np.array([obj['category_id'] for obj in annos])

            new_image = np.zeros((self.dataset_size, self.dataset_size, 3))
            new_seg = np.zeros((self.dataset_size, self.dataset_size))

            # reshape_image_size < dataset_size
            if new_h <= self.dataset_size and new_w <= self.dataset_size:
                local_h = random.randint(0, self.dataset_size - new_h)
                local_w = random.randint(0, self.dataset_size - new_w)
                new_image[local_h:local_h+new_h, local_w:local_w+new_w, :] += image_transformed
                new_seg[local_h:local_h+new_h, local_w:local_w+new_w] += seg_transformed
                bbox_transformed[:, [0, 2]] += local_w
                bbox_transformed[:, [1, 3]] += local_h

            # reshape_image_size > dataset_size
            else:
                res_h = self.dataset_size - new_h if (self.dataset_size - new_h > 0) else 0
                res_w = self.dataset_size - new_w if (self.dataset_size - new_w > 0) else 0
                offset_h = random.randint(0, res_h)
                offset_w = random.randint(0, res_w)
                temp_image = image_transformed[offset_h:offset_h + self.dataset_size, offset_w:offset_w + self.dataset_size, :]
                temp_seg = seg_transformed[offset_h:offset_h + self.dataset_size, offset_w:offset_w + self.dataset_size]
                new_image[:temp_image.shape[0], :temp_image.shape[1], :] += temp_image
                new_seg[:temp_seg.shape[0], :temp_seg.shape[1]] += temp_seg
                bbox_transformed[:, [0, 2]] -= offset_w
                bbox_transformed[:, [1, 3]] -= offset_h
                bbox_transformed = np.clip(bbox_transformed, 0, self.dataset_size)
                
                # some instance may not in image
                mask = (bbox_transformed[:, 2] - bbox_transformed[:, 0]) * (bbox_transformed[:, 3] - bbox_transformed[:, 1]) > 0
                bbox_transformed = bbox_transformed[mask]
                ins_cls = ins_cls[mask]
            return np.transpose(new_image, (2, 0, 1)), bbox_transformed, new_seg, ins_cls
        
        # ImageNet_data reshape
        else:
            augs = T.AugmentationList([
            T.RandomFlip(prob=0.5),
            T.ResizeTransform(h, w, 224, 224),
        ])

            input = T.AugInput(ori_input, sem_seg=seg, boxes=bbox)
            transform = augs(input)  # type: T.Transform
            image_transformed = input.image  # new image
            seg_transformed = input.sem_seg
            bbox_transformed = input.boxes
            
            # ？？
            new_image = image_transformed[int(bbox_transformed[0][1]):int(bbox_transformed[0][3] + 1), int(bbox_transformed[0][0]):int(bbox_transformed[0][2] + 1),:]
            new_seg = seg_transformed[int(bbox_transformed[0][1]):int(bbox_transformed[0][3] + 1), int(bbox_transformed[0][0]):int(bbox_transformed[0][2] + 1)]
            bbox_transformed[:,[0, 1]] = 0
            bbox_transformed[:,[2, 3]] = new_image.shape[:2]
            
            return np.transpose(new_image, (2, 0, 1)), bbox_transformed, new_seg

            
   
    def __call__(self, dataset_dict:dict, imageNet_dict: list) -> dict:
        dataset_dict = copy.deepcopy(dataset_dict)
        h, w = dataset_dict['height'], dataset_dict['width']

        if 'file_name' in dataset_dict:
            ori_image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format)
            
        
        utils.check_image_size(dataset_dict, ori_image)


        # ？dataset no sem_seg
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        
        # ？
        if self.is_debug:
            dataset_dict['dataset_source'] = 0

        not_full_labeled = 'dataset_source' in dataset_dict and \
            self.with_ann_type and \
                self.dataset_ann[dataset_dict['dataset_source']] != 'box'
                
        # get_bbox
        bbox = []
        seg = np.zeros((h, w))
        for i in range(len(dataset_dict['annotations'])):
            bbox.append(dataset_dict['annotations'][i]['bbox'])
            mask = np.ascontiguousarray(mask_util.decode(mask_util.frPyObjects(dataset_dict['annotations'][i]['segmentation'], h, w)))
            if len(mask.shape) > 2:
                mask = mask.transpose((2, 0, 1)).sum(0) * dataset_dict['annotations'][i]['category_id']
            seg += mask

        bbox = np.array(bbox)
        new_image, new_bbox, new_seg, ins_cls = self.reshape_image(ori_image, dataset_dict, seg)
        
        # copy and paste
        append_box = []
        append_cls = []
        for image_dict in imageNet_dict:
            copy_image = utils.read_image(
                image_dict['file_name'], format=self.image_format)
            
            utils.check_image_size(image_dict, copy_image)

            append_cls.append(image_dict['category_id'])

            seg = image_dict['segmentation']
            image_box = image_dict['bbox']
            copy_image, image_box, image_seg = self.reshape_image(ori_input=copy_image, seg=seg, bbox=image_box)

            # get mask and seg img
            image_seg[image_seg > 0] = 1
            mask = image_seg
            temp_image = copy_image * mask

            # get a random place to put
            tmp_y = random.randrange(max(int(self.dataset_size - image_seg.shape[0]), 1))
            tmp_x = random.randrange(max(int(self.dataset_size - image_seg.shape[1]), 1))

            # replace the original img and seg
            new_image[:, tmp_y:(tmp_y + image_seg.shape[0]), tmp_x:(tmp_x + image_seg.shape[1])] *= 1 - mask
            new_image[:, tmp_y:(tmp_y + image_seg.shape[0]), tmp_x:(tmp_x + image_seg.shape[1])] += temp_image

            image_box[:, [0, 2]] += tmp_x
            image_box[:, [1, 3]] += tmp_y
            append_box.append(image_box)
        
        append_box = np.asarray(append_box)
        
        # final info
        image_shape = new_image.shape[1:]
        # get Instaces
        bbox = np.concatenate(append_box, new_bbox)
        category = np.concatenate(append_cls, ins_cls)
        all_annos = []
        for i in range(len(bbox)):
            ann = {}
            ann['bbox'] = bbox[i]
            ann["bbox_mode"] = BoxMode.XYXY_ABS
            ann["category_id"] = category[i]
            all_annos.append(ann)
        
        instances = utils.annotations_to_instances(all_annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(new_image))
        # TODO
        # more info to add
        return dataset_dict