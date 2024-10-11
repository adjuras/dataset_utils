import json
import cv2
import os
import argparse


def visualize_coco_bbox(dataset_dir, frame_idx, image_size, split="train"):
    width,height = image_size
    frame_path = os.path.join(dataset_dir, "images", split, f"frame_{frame_idx}.jpg")
    bbox_path = os.path.join(dataset_dir, "labels", split, f"frame_{frame_idx}.txt")
    with open(bbox_path,"r") as fp:
        line = fp.readline()
        elems = line.split(" ")
        cx = float(elems[1])
        cy = float(elems[2])
        w = float(elems[3])
        h = float(elems[4])
        x1 = (cx-w/2)*width
        y1 = (cy-h/2)*height
        x2 = (cx+w/2)*width
        y2 = (cy+h/2)*height
        bbox = (int(x1), int(y1), int(x2), int(y2))

    frame = cv2.imread(frame_path)
    image_viz = cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:4]), (255,0,0))
    while True:
        cv2.imshow('Image', image_viz)
        # 0xFF is a bitmask used to only get the last 8 bits of the return value
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def visualize_dataset_bbox(dataset_dir, frame_idx):
    image = cv2.imread(os.path.join(dataset_dir, "frame_"+str(frame_idx)+".jpg"))
    with open(os.path.join(dataset_dir,"annotations.json"), "r") as fp:
        anns = json.load(fp)
    bbox = anns[str(frame_idx)]
    bbox = [int(b) for b in bbox]
    print(bbox)
    image_viz = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:4]), (255,0,0))
    while True:
        cv2.imshow('Image', image_viz)
        # 0xFF is a bitmask used to only get the last 8 bits of the return value
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=="__main__":
    parser = argparse.ArgumentParser("visualize bbox from dataset")
    parser.add_argument("--dataset_dir")
    parser.add_argument("--frame_idx", type=int)
    parser.add_argument("--image_width", type=int)
    parser.add_argument("--image_height", type=int)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    #visualize_dataset_bbox(args.dataset_dir, args.frame_idx)
    visualize_coco_bbox(args.dataset_dir, args.frame_idx, (args.image_width, args.image_height),args.split)
