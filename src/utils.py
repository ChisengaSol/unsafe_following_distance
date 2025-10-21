import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


def list_files_in_subfolders(base_dir, limit=5):
    """
    Walks through all subfolders in the base_dir and prints some
    number of filenames from each.
    """

    for root, dirs, files in os.walk(base_dir):
        if root == base_dir:
            continue

        print(f"\nFolder: {root}")
        files_to_show = files[:limit]

        if not files_to_show:
            print("(No files in this folder)")
        else:
            for filename in files_to_show:
                print(f"  -- {filename}")


def verify_image_label_pairs(image_dir, label_dir):
    """
    Checks to ensure every .jpg image has a corresponding .txt label file,
    ignoring the 'classes.txt' file.
    """
    try:
        image_basenames = {
            os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")
        }
        label_basenames = {
            os.path.splitext(f)[0]
            for f in os.listdir(label_dir)
            if f.endswith(".txt") and f != "classes.txt"
        }
    except FileNotFoundError as e:
        print(f"Error: Directory not found. {e}")
        return

    print(f"Total images (.jpg) found: {len(image_basenames)}")
    print(f"Total annotation files (.txt) found: {len(label_basenames)}")

    matching_files = image_basenames.intersection(label_basenames)
    images_without_labels = image_basenames.difference(label_basenames)
    labels_without_images = label_basenames.difference(image_basenames)

    print(f"Matching image/label pairs: {len(matching_files)}")
    print(f"Images without labels: {len(images_without_labels)}")
    print(f"Labels without images: {len(labels_without_images)}")

    if not images_without_labels and len(labels_without_images) == 0:
        print("All images have a corresponding label file.")


def _convert_yolo_to_pixels(x_center, y_center, w, h, img_w, img_h):
    """Helper function to convert YOLO format to pixel coordinates."""
    x_center_abs = x_center * img_w
    y_center_abs = y_center * img_h
    w_abs = w * img_w
    h_abs = h * img_h
    x1 = int(x_center_abs - (w_abs / 2))
    y1 = int(y_center_abs - (h_abs / 2))
    x2 = int(x_center_abs + (w_abs / 2))
    y2 = int(y_center_abs + (h_abs / 2))
    return x1, y1, x2, y2


def visualize_images(image_dir, label_dir, image_filenames, class_names, num_images=5):
    """
    Loads original images and labels, draws bounding boxes, and displays them.
    """
    if len(image_filenames) < num_images:
        num_images = len(image_filenames)

    images_to_display = random.sample(image_filenames, num_images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
    if num_images == 1:
        axes = [axes]

    for i, image_name in enumerate(images_to_display):
        image_path = os.path.join(image_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape

        ax = axes[i]

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x, y, w, h = [float(p) for p in parts]
                    x1, y1, x2, y2 = _convert_yolo_to_pixels(x, y, w, h, img_w, img_h)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        class_names[int(class_id)],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

        ax.imshow(image)
        ax.set_title(image_name, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_augmented_batch(data_loader, class_names, num_images=4):
    """
    Fetches one batch from the data_loader and displays it in a horizontal grid.
    """
    batch_size = data_loader.batch_size
    if num_images > batch_size:
        num_images = batch_size

    images, targets = next(iter(data_loader))

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        image = images[i].permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        vis_image = image.copy()

        boxes = targets[i]["boxes"]
        labels = targets[i]["labels"]
        height, width, _ = vis_image.shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height

        for box, label_idx in zip(boxes, labels):
            xmin, ymin, xmax, ymax = map(int, box)
            label_name = class_names[label_idx]
            cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                vis_image,
                label_name,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        axes[i].imshow(vis_image)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
