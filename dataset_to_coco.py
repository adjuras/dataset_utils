import argparse
import os
import json
from tqdm import tqdm
import shutil

def bbox_to_yolo(bbox, image_width, image_height):
    """
    Convert bounding box from (x1, y1, x2, y2) format to YOLO format.

    Args:
        x1 (float): Top-left x-coordinate of the bounding box.
        y1 (float): Top-left y-coordinate of the bounding box.
        x2 (float): Bottom-right x-coordinate of the bounding box.
        y2 (float): Bottom-right y-coordinate of the bounding box.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        list: Bounding box in YOLO format [x_center, y_center, width, height] (normalized).
    """
    # Calculate the center of the bounding box
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Calculate the width and height of the bounding box
    width = x2 - x1
    height = y2 - y1

    # Normalize the values by the dimensions of the image
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    # Return the bounding box in YOLO format
    return [x_center, y_center, width, height]

def export_annotations_coco(annotations_path, image_width, image_height, output_path):
    '''
        Reads annotations.json file {idx : [x1,y1,x2,y2]} and exports coco annotations
        frame_{idx}.txt for each entry
    '''
    print("Exporting annotations")
    with open(annotations_path, "r") as fp:
        anns_dict = json.load(fp)
    for idx, bbox in tqdm(anns_dict.items()):
        bbox_yolo = bbox_to_yolo(bbox, image_width, image_height)
        export_path = os.path.join(output_path, f"frame_{idx}.txt")
        # Save .txt label file
        with open(export_path, "w") as fpp:
            fpp.write(f"0 {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}")

def copy_images(dataset_dir, output_path):
    print("Copying images")
    image_names = [img for img in os.listdir(dataset_dir) if ".jpg" in img]
    for img in tqdm(image_names):
        shutil.copyfile(os.path.join(dataset_dir,img), os.path.join(output_path,img))

def convert_dataset(dataset_dir, image_width, image_height):
    top_dir_path = os.path.join(dataset_dir, "coco")
    if not os.path.exists(top_dir_path):
        os.mkdir(top_dir_path)
    labels_path = os.path.join(top_dir_path, "labels")
    if not os.path.exists(labels_path):
        os.mkdir(labels_path)
    images_path = os.path.join(top_dir_path, "images")
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    # Save .txt label file
    export_annotations_coco(os.path.join(dataset_dir, "annotations.json"), image_width, image_height, labels_path)
    copy_images(dataset_dir, images_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Converts dataset to COCO format")
    parser.add_argument("--dataset_dir")
    parser.add_argument("--image_width", type=int)
    parser.add_argument("--image_height", type=int)
    args = parser.parse_args()
    convert_dataset(args.dataset_dir, args.image_width, args.image_height)
