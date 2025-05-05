import os
import cv2
import numpy as np
from config import INPUT_DIR, OUTPUT_DIR, SCALE_FACTOR, BRIGHTNESS, ROTATION_ANGLES, FLIP_MODES
from HandTrackingModule import handDetector

def apply_augmentations(image):
    augmented_images = []
    h, w = image.shape[:2]

    """for scale in SCALE_FACTOR:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        scaled = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(scaled)"""

    for alpha in BRIGHTNESS:
        bright = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented_images.append(bright)

    for angle in ROTATION_ANGLES:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)

    for flip_mode in FLIP_MODES:
        flipped = cv2.flip(image, flip_mode)
        augmented_images.append(flipped)

    return augmented_images

def preprocess_and_save():
    detector = handDetector(detectionCon=0.7)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for split in ['train', 'test', 'validation']:
        split_input_path = os.path.join(INPUT_DIR, split)
        split_output_path = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_output_path, exist_ok=True)

        for class_folder in os.listdir(split_input_path):
            input_class_path = os.path.join(split_input_path, class_folder)
            output_class_path = os.path.join(split_output_path, class_folder)
            os.makedirs(output_class_path, exist_ok=True)

            for img_name in os.listdir(input_class_path):
                img_path = os.path.join(input_class_path, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue

                img = detector.findHands(img, draw=False)
                lmList = detector.findPosition(img, draw=False)

                if len(lmList) != 0:
                    x_list = [pt[1] for pt in lmList]
                    y_list = [pt[2] for pt in lmList]
                    x_min, x_max = max(min(x_list) - 20, 0), min(max(x_list) + 20, img.shape[1])
                    y_min, y_max = max(min(y_list) - 20, 0), min(max(y_list) + 20, img.shape[0])

                    roi = img[y_min:y_max, x_min:x_max]
                    if roi.size != 0:
                        roi_resized = cv2.resize(roi, (64, 64))
                        save_path = os.path.join(output_class_path, img_name)
                        cv2.imwrite(save_path, roi_resized)

                        augmented = apply_augmentations(roi_resized)
                        for i, aug_img in enumerate(augmented):
                            aug_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
                            cv2.imwrite(os.path.join(output_class_path, aug_name), aug_img)
                else:
                    print(f"No hand detected in {img_path}")

    print("Preprocessing with hand detection completed.")
