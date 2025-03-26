import os
import sys
import cv2
import random
import numpy as np
from math import sin, cos, tan, ceil, pi
from pathlib import Path
from tqdm import tqdm
import uuid

def crop_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    while True:
        top, bottom = np.random.normal(h // 12, h // 12, 2).tolist()
        left, right = np.random.normal(w // 12, w // 12, 2).tolist()

        top = min(h // 4, max(0, round(top)))
        bottom = min(h // 4, max(0, round(bottom)))
        left = min(w // 4, max(0, round(left)))
        right = min(w // 4, max(0, round(right)))

        # Avoid aspect ratio too extreme
        new_h = h - top - bottom
        new_w = w - left - right
        if max(new_h, new_w) / min(new_h, new_w) <= 2: break
    
    return image[top:h-bottom, left:w-right, :]

def rotate_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    angle = random.randint(-15, 15)

    if angle == 0: return image

    rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h))
    
    radian = abs(angle) / 180 * pi
    x = ((h - w * (1/sin(radian) + 1/tan(radian))) 
         / (-(1/cos(radian) + 1) * (1/sin(radian) + 1/tan(radian)) + tan(radian)))
    y = w - (1/cos(radian) + 1) * x
    x = ceil(x * tan(radian))
    y = ceil(y)

    return rotated[x:-x, y:-y, :]

def horizontal_flip_image(image: np.ndarray) -> np.ndarray:
    if random.randint(0, 1):
        return image[:, ::-1, :]
    return image

def blur_image(image: np.ndarray) -> np.ndarray:
    w = random.randint(1, 3)
    return cv2.blur(image, (w, w))

def bright_image(image: np.ndarray) -> np.ndarray:
    beta = random.randint(-30, 30)
    return cv2.convertScaleAbs(image, beta=beta)

def generate_random_image(image_dir: str):
    exist = set(Path(f).stem for f in os.listdir(image_dir))
    not_augmented = [f for f in exist if not f.startswith('aug_')]
    N = min(400, len(not_augmented) * 3)

    for _ in tqdm(range(N - len(exist))):
        file_name = random.choice(not_augmented)
        new_file_name = f'aug_{str(uuid.uuid4())}'

        image = cv2.imread(f'{image_dir}/{file_name}.jpg')

        image = crop_image(image)
        image = rotate_image(image)
        image = horizontal_flip_image(image)
        image = blur_image(image)
        image = bright_image(image)

        cv2.imwrite(f'{image_dir}/{new_file_name}.jpg', image)

def main():
    if len(sys.argv) != 2:
        print('Usage: data_augment.py <training data root directory>')
        exit(1)
    
    train_dir = sys.argv[1]

    for class_ in range(100):
        image_dir = f'{train_dir}/{class_}'

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f'Directory for class {class_} not exists')

        generate_random_image(image_dir)

        print(f'Class {class_} finish')

if __name__ == '__main__': 
    main()