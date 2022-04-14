import cv2, os

def load_images_from_folder(folder):
    images = []
    files = os.listdir(folder)
    files.sort()
    for filename in files:
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img, (512, 512))
        if img is not None:
            images.append(img)
    return files, images


def load_dataset(dataset_path):
    images_path = dataset_path+'images/'
    annotation_path = dataset_path+'annotation/'

    img_names, img_list = load_images_from_folder(images_path)
    msk_names, msk_list = load_images_from_folder(annotation_path)


    x = np.asarray(img_list, dtype=np.float32)/255
    y = np.asarray(msk_list, dtype=np.float32)/255
    y[y>0] = 1
    y = y.sum(axis=3).reshape(47, y.shape[1], y.shape[2], 1)

    return x, y

def get_augmented(x, y):
    from keras_unet.utils import get_augmented
    train_gen = get_augmented(
        x, y, batch_size=2,
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