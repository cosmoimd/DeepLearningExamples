# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import torch
import random
from torch.utils.data import DataLoader

from ssd.utils import dboxes300_coco, COCODetection
from ssd.utils import SSDTransformer
from pycocotools.coco import COCO
# DALI import
from ssd.coco_pipeline import COCOPipeline, DALICOCOIterator


def get_train_loader(args, local_seed):
    if args.dataset_name == 'real_colon':
        train_annotate = os.path.join(args.data, "train_ann.json")
        train_coco_root = os.path.join(args.data, "train_images")
    else:
        train_annotate = os.path.join(args.data, "annotations/instances_train2017.json")
        train_coco_root = os.path.join(args.data, "train2017")

    if args.dataset_name == 'real_colon' and args.negatives_sampling:
        # Load original COCO annotations
        with open(train_annotate) as f:
            json_data = json.load(f)

        # Separate images with and without annotations
        images_with_annotations = set()
        for ann in json_data['annotations']:
            images_with_annotations.add(ann['image_id'])
        all_images = {img['id']: img for img in json_data['images']}
        empty_images = [img for img_id, img in all_images.items() if img_id not in images_with_annotations]
        images_with_annotations = [img for img_id, img in all_images.items() if img_id in images_with_annotations]

        # Randomly sample the same number of negative images and positive images
        num_to_sample = len(images_with_annotations)
        sampled_empty_images = random.sample(empty_images, num_to_sample)
        json_data['images'] = images_with_annotations + sampled_empty_images

        train_annotate = os.path.join(args.save, "train_temp.json")
        with open(train_annotate, 'w') as f:
            json.dump(json_data, f)

        num_train_images = len(json_data['images'])
    else:
        num_train_images = len(os.listdir(train_coco_root))
    print(f"Setting epoch size to {num_train_images}, 'skip_empty': {args.skip_empty}")

    train_pipe = COCOPipeline(batch_size=args.batch_size,
                              file_root=train_coco_root,
                              annotations_file=train_annotate,
                              default_boxes=dboxes300_coco(),
                              device_id=args.local_rank,
                              num_shards=args.N_gpu,
                              output_fp16=args.amp,
                              output_nhwc=False,
                              pad_output=False,
                              num_threads=args.num_workers,
                              seed=local_seed,
                              skip_empty=args.skip_empty)
    train_pipe.build()
    train_loader = DALICOCOIterator(train_pipe, num_train_images / args.N_gpu)

    return train_loader


def get_val_dataset(args):
    dboxes = dboxes300_coco()
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)

    if args.dataset_name == 'real_colon':
        val_annotate = os.path.join(args.data, "validation_ann.json")
        val_coco_root = os.path.join(args.data, "validation_images")
    else:
        val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(args.data, "val2017")

    val_coco = COCODetection(val_coco_root, val_annotate, val_trans, skip_empty=args.skip_empty)
    return val_coco


def get_test_dataset(args):
    dboxes = dboxes300_coco()
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)
    if args.dataset_name == 'real_colon':
        val_annotate = os.path.join(args.data, "test_ann.json")
        val_coco_root = os.path.join(args.data, "test_images")
    else:
        val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(args.data, "val2017")

    val_coco = COCODetection(val_coco_root, val_annotate, val_trans, skip_empty=args.skip_empty)
    return val_coco


def get_val_dataloader(dataset, args):
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        val_sampler = None

    val_dataloader = DataLoader(dataset,
                                batch_size=args.eval_batch_size,
                                shuffle=False,  # Note: distributed sampler is shuffled :(
                                sampler=val_sampler,
                                num_workers=args.num_workers)

    return val_dataloader


def get_coco_ground_truth_validation(args):
    val_annotate = None
    print(args.data)
    if args.dataset_name == 'real_colon':
        val_annotate = os.path.join(args.data, "validation_ann.json")
    else:
        val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")

    cocoGt = COCO(annotation_file=val_annotate, use_ext=True)
    return cocoGt


def get_coco_ground_truth_test(args):
    if args.dataset_name == 'real_colon':
        test_annotate = os.path.join(args.data, "test_ann.json")
    else:
        test_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    cocoGt = COCO(annotation_file=test_annotate, use_ext=True)
    return cocoGt
