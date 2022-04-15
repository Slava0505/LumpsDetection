import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    "Create list of images from folder directory"
    images = []
    files = os.listdir(folder)
    files.sort()
    for filename in files:
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img, (512, 512))
        if img is not None:
            images.append(img)
    return images


def load_dataset(dataset_path):
    "Load all images and annotation list from dataset folder"
    images_path = dataset_path+'images/'
    annotation_path = dataset_path+'annotation/'

    img_list = load_images_from_folder(images_path)
    msk_list = load_images_from_folder(annotation_path)


    x = np.asarray(img_list, dtype=np.float32)/255
    y = np.asarray(msk_list, dtype=np.float32)/255
    y[y>0] = 1
    y = y.sum(axis=3).reshape(y.shape[0], y.shape[1], y.shape[2], 1)

    return x, y

def get_augmented_data_loader(dataset_path):
    "Load dataset and apply augmentation and create generator from list of images and annotation"
    x, y = load_dataset(dataset_path)

    from keras_unet.utils import get_augmented

    train_gen = get_augmented(
        x,
        y, 
        batch_size=2,
        data_gen_args = dict(
            rotation_range=5.,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=40,
            zoom_range=0.0,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='constant'
        ))
    return train_gen


def load_demo_dataset(dataset_path):
    "Load video to image list"

    cap = cv2.VideoCapture(demo_path)
    x = []
    ret=True
    while(ret):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (512, 512))
            x.append(frame)

    x = np.asarray(x, dtype=np.float32)/255

    return x


def mask_to_rgba(mask, color="red"):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    
    Args:
        mask (numpy.ndarray): [description]
        color (str, optional): Check `MASK_COLORS` for available colors. Defaults to "red".
    
    Returns:
        numpy.ndarray: [description]
    """    
    assert(color in [
    "red", "green", "blue",
    "yellow", "magenta", "cyan"
])
    assert(mask.ndim==3 or mask.ndim==2)

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones), axis=-1)
def zero_pad_mask(mask, desired_size):
    """[summary]
    
    Args:
        mask (numpy.ndarray): [description]
        desired_size ([type]): [description]
    
    Returns:
        numpy.ndarray: [description]
    """
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask