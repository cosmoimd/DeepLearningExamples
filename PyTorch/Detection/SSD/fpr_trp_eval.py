#!/usr/bin/env python

""" Compute False Positive and True Positive Rates from output jsons folder

    Usage:
        - python3 fpr_trp_eval.py <path_to_json_folder>

    Copyright 2023-, Cosmo Intelligent Medical Devices
"""
import json
import os
import sys


def fpr_tpr(base_path):
    """
    Compute False Positive and True Positive Rates from output jsons folder.

    Args:
        base_path (str): The folder path where annotation JSON files are located.

    Returns:
    """
    # Specify the path to the JSON files saved by SSD evaluation code
    json_results_path = os.path.join(base_path, "predictions.json")
    json_gt_path = os.path.join(base_path, "ground_truth.json")

    threshold = 0.5 # threshold on prediction class scores

    # Load the JSON file
    with open(json_results_path, "r") as json_file:
        prediction_results = json.load(json_file)
    print(f"Opened prediction file {json_results_path}")
    with open(json_gt_path, "r") as json_file:
        gt_file = json.load(json_file)
    print(f"Opened ground truth file {json_gt_path}")

    id_of_negative_frames = []
    id_of_frames_with_predictions = []
    id_of_positive_frames = []
    for gt in gt_file:
        if len(gt['annotations']) == 0:
            id_of_negative_frames.append(gt['id'])
        else:
            id_of_positive_frames.append(gt['id'])

    for pred in prediction_results:
        if pred["score"] > threshold:
            id_of_frames_with_predictions.append(pred['id'])

    # Calculating FPR
    set1 = set(id_of_negative_frames)
    set2 = set(id_of_frames_with_predictions)
    intersection = set1 & set2
    print("False Positive Rate: ", len(intersection) / len(id_of_negative_frames))

    # Calculating TPR
    true_positives = [pred for pred in prediction_results if pred["score"] > threshold and
                      pred['id'] in id_of_positive_frames]
    tpr = len(true_positives) / len(id_of_positive_frames) if id_of_positive_frames else 0
    print("True Positive Rate: ", tpr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: fpr_trp_eval.py <path_to_json_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]
    fpr_tpr(folder_path)
