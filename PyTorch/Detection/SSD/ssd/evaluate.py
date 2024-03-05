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

import torch
import time
import numpy as np
from contextlib import redirect_stdout
import io
import json
import os

from pycocotools.cocoeval import COCOeval


def evaluate(model, coco, cocoGt, encoder, inv_map, args):
    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    model.eval()
    if not args.no_cuda:
        model.cuda()
    ret = []
    start = time.time()

    # for idx, image_id in enumerate(coco.img_keys):
    for nbatch, (img, img_id, img_size, _, _) in enumerate(coco):
        print("Parsing batch: {}/{}".format(nbatch, len(coco)), end='\r')
        with torch.no_grad():
            inp = img.cuda()
            with torch.cuda.amp.autocast(enabled=args.amp):
                # Get predictions
                ploc, plabel = model(inp)
            ploc, plabel = ploc.float(), plabel.float()

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            for idx in range(ploc.shape[0]):
                # ease-of-use for specific predictions
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, 0.50, 200)[0]
                except Exception as e:
                    print("Skipping idx {}, failed to decode with message {}, Skipping.".format(idx, e))
                    continue

                htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id[idx], loc_[0] * wtot, \
                                loc_[1] * htot,
                                (loc_[2] - loc_[0]) * wtot,
                                (loc_[3] - loc_[1]) * htot,
                                prob_,
                                inv_map[label_]])

    # Now we have all predictions from this rank, gather them all together
    # if necessary
    ret = np.array(ret).astype(np.float32)

    # Multi-GPU eval
    if args.distributed:
        # NCCL backend means we can only operate on GPU tensors
        ret_copy = torch.tensor(ret).cuda()
        # Everyone exchanges the size of their results
        ret_sizes = [torch.tensor(0).cuda() for _ in range(N_gpu)]

        torch.cuda.synchronize()
        torch.distributed.all_gather(ret_sizes, torch.tensor(ret_copy.shape[0]).cuda())
        torch.cuda.synchronize()

        # Get the maximum results size, as all tensors must be the same shape for
        # the all_gather call we need to make
        max_size = 0
        sizes = []
        for s in ret_sizes:
            max_size = max(max_size, s.item())
            sizes.append(s.item())

        # Need to pad my output to max_size in order to use in all_gather
        ret_pad = torch.cat([ret_copy, torch.zeros(max_size - ret_copy.shape[0], 7, dtype=torch.float32).cuda()])

        # allocate storage for results from all other processes
        other_ret = [torch.zeros(max_size, 7, dtype=torch.float32).cuda() for i in range(N_gpu)]
        # Everyone exchanges (padded) results

        torch.cuda.synchronize()
        torch.distributed.all_gather(other_ret, ret_pad)
        torch.cuda.synchronize()

        # Now need to reconstruct the _actual_ results from the padded set using slices.
        cat_tensors = []
        for i in range(N_gpu):
            cat_tensors.append(other_ret[i][:sizes[i]][:])

        final_results = torch.cat(cat_tensors).cpu().numpy()
    else:
        # Otherwise full results are just our results
        final_results = ret

    if args.local_rank == 0:
        print("")
        print("Predicting Ended, total time: {:.2f} s".format(time.time() - start))

    # Create jsons with per-frame predictions and ground truths to compute whole-video statistics
    if args.dataset_name == "real_colon":

        json_start_time = time.time()
        # Specify the path where you want to save the JSON file
        output_result_path = os.path.join(args.inference_jsons, "predictions.json")
        output_ground_truth_path = os.path.join(args.inference_jsons, "ground_truth.json")
        print("Creating json file for predictions...")

        # Convert final_results and cocoGt.dataset to a serializable format
        serializable_results = []
        serializable_gts = []
        # Populate serializable_results
        for result in final_results:
            serializable_result = {
                "id": int(result[0]),
                "bbox": [float(result[1]), float(result[2]),
                         float(result[1]) + float(result[3]),
                         float(result[2]) + float(result[4])],
                "score": float(result[5]),
                "label": int(result[6]),
            }
            serializable_results.append(serializable_result)

        # Save to a JSON file
        with open(output_result_path, "w") as json_file:
            json.dump(serializable_results, json_file)
        print(f"Created predictions json file of size {len(serializable_results)}.")
        print("Creating json file for ground truth...")

        # Populate serializable_gt
        for image_info in cocoGt.dataset['images']:
            img_id = image_info['id']
            annotations_list = []

            # Collect all annotations for the current image
            for annotation in cocoGt.dataset['annotations']:
                if annotation['image_id'] == img_id:
                    bbox_converted = [annotation['bbox'][0], annotation['bbox'][1],
                                      annotation['bbox'][0] + annotation['bbox'][2],
                                      annotation['bbox'][1] + annotation['bbox'][3]]
                    annotations_list.append({
                        "unique_id": annotation['unique_id'],
                        "bbox": bbox_converted,
                        "label": 1  # or the actual label if available
                    })

            serializable_gt = {
                "id": img_id,
                "file_name": image_info['file_name'],
                "annotations": annotations_list
            }

            serializable_gts.append(serializable_gt)

        # Save to a JSON file
        with open(output_ground_truth_path, "w") as json_file:
            json.dump(serializable_gts, json_file)
        print(f"Created ground truth json file of size {len(serializable_gts)}")

        json_end_time = time.time()
        print(f"Finished creating jsons in {json_end_time - json_start_time} seconds.")

    cocoDt = cocoGt.loadRes(final_results, use_ext=True)

    E = COCOeval(cocoGt, cocoDt, iouType='bbox', use_ext=True)
    E.evaluate()
    E.accumulate()
    if args.local_rank == 0:
        E.summarize()
        print("Current AP: {:.5f}".format(E.stats[0]))
    else:
        # fix for cocoeval indiscriminate prints
        with redirect_stdout(io.StringIO()):
            E.summarize()

    # put your model in training mode back on
    model.train()

    return E.stats[0]  # Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]

