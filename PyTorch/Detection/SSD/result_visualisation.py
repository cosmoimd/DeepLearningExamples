#!/usr/bin/env python

""" Starting from inference output jsons folder, produce videos with model predictions and GT boxes to
   assess model performance.

    Usage:
        - python3 result_visualisation.py <path_to_json_folder> <output_folder>

    Copyright 2023-, Cosmo Intelligent Medical Devices
"""
import json
import sys
import cv2
import os
from PIL import Image, ImageDraw


def rescale_bbox(bbox, orig_width, orig_height, target_width, target_height):
    """
    Rescale bounding box coordinates to fit a new image size.

    Args:
        bbox (list): List containing bounding box coordinates in the format [x_min, y_min, x_max, y_max].
        orig_width (int): Original width of the image.
        orig_height (int): Original height of the image.
        target_width (int): Target width to which the bounding box coordinates will be rescaled.
        target_height (int): Target height to which the bounding box coordinates will be rescaled.

    Returns:
        list: Rescaled bounding box coordinates in the format [x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled].
    """
    # Rescale the bounding box coordinates
    x_min, y_min, x_max, y_max = bbox
    x_min_scaled = int((x_min / orig_width) * target_width)
    y_min_scaled = int((y_min / orig_height) * target_height)
    x_max_scaled = int((x_max / orig_width) * target_width)
    y_max_scaled = int((y_max / orig_height) * target_height)
    return [x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled]


def visual(base_path, videos_output_folder):
    """
    Compute False Positive and True Positive Rates from output jsons folder.

    Args:
        base_path (str): The folder path where annotation JSON files are located.
        videos_output_folder (str): The folder path where to save output videos.

    Returns:
    """
    # Specify the path to the JSON files
    json_results_path = os.path.join(base_path, "predictions.json")
    json_gt_path = os.path.join(base_path, "ground_truth.json")

    # Load the JSON file
    with open(json_results_path, "r") as json_file:
        prediction_results = json.load(json_file)
    print(f"Opened prediction file {json_results_path}")
    with open(json_gt_path, "r") as json_file:
        gt_file = json.load(json_file)
    print(f"Opened ground truth file {json_gt_path}")

    # Specify the test videos here
    video_names = ["001-013", "002-013", "003-013", "004-013",
                   "001-014", "002-014", "003-014", "004-014",
                   "001-015", "002-015", "003-015", "004-015"]
    for vn in video_names:
        new_gt_file = []
        for gt in gt_file:
            if vn in gt["file_name"]:
                new_gt_file.append(gt)
            max_id = gt["id"]

        new_pred_file = []
        for pred in prediction_results:
            if pred["id"] <= max_id:
                new_pred_file.append(pred)

        # Convert gt_file into a dictionary for faster access
        gt_dict = {item['id']: item for item in new_gt_file}

        # Iterate over prediction_results and add the predictions to the corresponding new_gt_file entry
        for prediction in new_pred_file:
            image_id = prediction['id']
            # Check if the image_id exists in gt_dict (new_gt_file)
            if image_id in gt_dict:
                # If the 'preds' key does not exist, create it
                if 'preds' not in gt_dict[image_id]:
                    gt_dict[image_id]['preds'] = []

                if prediction['score'] > 0.5:
                    # Add the bbox prediction to the 'annotations' list
                    gt_dict[image_id]['preds'].append(prediction['bbox'])

        # Optionally, convert gt_dict back to a list if needed for further processing
        updated_new_gt_file = list(gt_dict.values())

        def extract_frame_number(entry):
            # Assuming file_name format is always like '001-013_framenumber.jpg'
            # Split by '_' then take the second part and split by '.' to remove the extension,
            # finally convert to integer
            _, frame_str = entry['file_name'].split('_')
            frame_number = int(frame_str.split('.')[0])
            return frame_number

        # Sort the updated_new_gt_file list by frame number
        updated_new_gt_file = sorted(updated_new_gt_file, key=extract_frame_number)

        # Drawing, frame per frame
        base_image_path = videos_output_folder + "/test_images/"
        temp_dir = videos_output_folder + "/temp_images_" + vn  # Temporary directory to save annotated images
        os.makedirs(temp_dir, exist_ok=True)
        for gt in updated_new_gt_file:
            file_name = gt["file_name"]
            image_path = os.path.join(base_image_path, file_name)

            # Load and rescale the image
            image = Image.open(image_path)
            orig_width, orig_height = image.size
            image_rescaled = image.resize((300, 300))

            # Create a copy for drawing
            draw = image_rescaled.copy()
            draw_draw = ImageDraw.Draw(draw)

            # Rescale and draw the bounding boxes
            for annotation in gt.get("annotations", []):
                bbox_scaled = rescale_bbox(annotation, orig_width, orig_height, 300, 300)
                draw_draw.rectangle(bbox_scaled, outline="green", width=5)

            for predn in gt.get("preds", []):
                bbox_scaled = rescale_bbox(predn, orig_width, orig_height, 300, 300)
                draw_draw.rectangle(bbox_scaled, outline="red", width=5)

            # Save the annotated image
            temp_image_path = os.path.join(temp_dir, file_name)
            draw.save(temp_image_path)

        # Step 4: Create a video from the annotated images
        video_name = videos_output_folder + vn + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, 20.0, (300, 300))

        for file_name in sorted(os.listdir(temp_dir)):
            img_path = os.path.join(temp_dir, file_name)
            img = cv2.imread(img_path)
            video.write(img)

        video.release()
        print(f"Video saved as {video_name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: per_video_eval.py <path_to_json_folder> <output_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_folder = sys.argv[2]
    visual(folder_path, output_folder)
