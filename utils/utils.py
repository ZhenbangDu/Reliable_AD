from diffusers import DPMSolverSinglestepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DDIMScheduler
from PIL import Image
import numpy as np
import cv2
import os

class CannyDetector(object):
    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    def __call__(self, input_image):
        return cv2.Canny(input_image, self.low_threshold, self.high_threshold)
        
def resize_and_canny(trans_path, preprocessor, scale, width, height, keep_loc, matting=False):
     init_images, mask_images, edge_images, post_masks=list(), list(), list(), list()
     for img in trans_path:
        file_name, ext = os.path.splitext(img)
        if ext not in ['.png', '.jpg']:
            continue
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if keep_loc:
            image = resize_image(image, width)
        else:
            image = golden_ratio_center_zoom(image, (width, height), scale, matting)
        mask = image[:, :, -1]
        matrix = 255 - np.asarray(mask).astype(np.uint8) 
        post_masks.append(Image.fromarray(matrix))
        kernel = np.ones((20, 20), np.uint8)
        matrix = cv2.dilate(matrix, kernel, 1)
        mask_images.append(Image.fromarray(matrix))  
        
        white = image[:, :, :3].copy()
        white[mask / 255.0 < 0.5] = 0
        edge_image = preprocessor(np.asarray(white).astype(np.uint8))
        edge_image = edge_image[:, :, None]
        edge_image = np.concatenate([edge_image, edge_image, edge_image], axis=2)
        edge_image = Image.fromarray(edge_image)
        edge_images.append(edge_image) 
        rgb_image = image[:, :, :3].copy()
        init_images.append(Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)))
     return init_images, mask_images, edge_images, post_masks

def choose_scheduler(scheduler_name):
    scheduler_dict = {
        'DPM++ SDE Karras': DPMSolverSinglestepScheduler,
        'Euler a': EulerAncestralDiscreteScheduler,
        'Euler': EulerDiscreteScheduler,
        'DDIM': DDIMScheduler,
    }
    if scheduler_name in scheduler_dict:
        print("="*20)
        return scheduler_dict[scheduler_name]
    return DPMSolverSinglestepScheduler

def resize_image(input_image, resolution):
    if input_image.shape[-1] == 3:
        input_image = white_img_add_mask(input_image)
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def white_img_add_mask(img):
    mask = np.expand_dims(np.uint8(np.where(img == [255, 255, 255], 0, 1).all(axis=2)) * 255, 2)
    new_img = np.concatenate([img, mask], axis=2)
    return new_img


def golden_ratio_center_zoom(src_image, input_size, scale=0.7, matting=False):
    if src_image.shape[-1] == 3:
        src_image = white_img_add_mask(src_image)

    W, H = input_size

    mask = src_image[:, :, -1]
    binary_mask = np.uint8(mask / 255.0 >= 0.5)

    x_sum = np.squeeze(np.sum(binary_mask, axis=0))
    
    if len(np.squeeze(np.nonzero(x_sum))) == 0:
        x1, x2 = 0, len(x_sum)
    else:
        x1 = np.squeeze(np.nonzero(x_sum))[0]   
        x2 = np.squeeze(np.nonzero(x_sum))[-1]

    y_sum = np.squeeze(np.sum(binary_mask, axis=1))

    if len(np.squeeze(np.nonzero(y_sum))) == 0:
        y1, y2 = 0, len(y_sum)
    else:
        y1 = np.squeeze(np.nonzero(y_sum))[0]
        y2 = np.squeeze(np.nonzero(y_sum))[-1]

    if x1 == x2 or y1 == y2:
        x1, x2 = 0, len(x_sum)
        y1, y2 = 0, len(y_sum)

    if matting == True:
        if x1 == 0 or x2 == src_image.shape[1] or y1 == 0 or src_image.shape[0] == y2:
            src_image = cv2.resize(src_image, (W, H), interpolation=cv2.INTER_AREA)
            return src_image
        
    img_crop = src_image[y1:y2, x1:x2, :]

    max_h = int(H * scale)
    max_w = int(W * scale)

    crop_h, crop_w = img_crop.shape[:2]
    if crop_h >= crop_w:
        new_h = np.minimum(max_h, int(max_w / crop_w * crop_h))
        new_w = int(new_h / crop_h * crop_w)
    else:
        new_w = np.minimum(max_w, int(max_h / crop_h * crop_w))
        new_h = int(new_w / crop_w * crop_h)
    new_crop = cv2.resize(img_crop, (new_w, new_h), interpolation = cv2.INTER_AREA)

    golden_center_y = H // 2
    beg_x = np.maximum(0, (W - new_w) // 2)
    beg_y = int(golden_center_y - new_h / 2)
    out_img = np.zeros((H, W, 4))
    out_img[beg_y:beg_y + new_h, beg_x:beg_x + new_w, :] = new_crop
    return np.uint8(out_img)

def concat_image(trans_path, image_path, save_folder):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    trans_image = cv2.imread(trans_path, cv2.IMREAD_UNCHANGED)
    if trans_image is not None:
        if trans_image.shape[-1] == 4:
            rgb = trans_image[:, :, :-1]
        else:
            rgb = trans_image
        mask = trans_image[:, :, -1]
        white = rgb.copy()
        white[mask / 255.0 < 0.5] = 255

        white = cv2.resize(white, (512, 512), interpolation = cv2.INTER_AREA)
        image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
        save_path = os.path.join(save_folder, os.path.basename(image_path))

        concat_image = np.concatenate([white, image], axis = 1)
        cv2.imwrite(save_path, concat_image)
        
    return concat_image