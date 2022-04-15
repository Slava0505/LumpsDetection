from keras_unet.models import custom_unet
from tensorflow.keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded
import numpy as np
import cv2

class CustomUnetMpdel():
    """Unet model for Lumb Segmentation
    
    Attributes
    ----------
    input_shape : int
        3D Tensor of shape (x, y, num_channels)
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape


    def initial_model(self):
        self.model = custom_unet(
            self.input_shape,
            filters=32,
            use_batch_norm=True,
            dropout=0.3,
            dropout_change_per_layer=0.0,
            num_layers=4
        )
        self.model.summary()
        self.model.compile(
            optimizer=Adam(), 
            loss='binary_crossentropy'
        )

    def train(self, dataset_path):
        from utils import get_augmented_data_loader
        train_gen = get_augmented_data_loader(dataset_path)

        self.initial_model()
        self.model.fit_generator(
            train_gen,
            steps_per_epoch=200,
            epochs=8
        )

    def evaluate(self, dataset_path):
        pass

    def demo(self, demo_path, out_folder_path):
        from utils import load_demo_dataset, mask_to_rgba, zero_pad_mask
        x_test = load_demo_dataset(demo_path)
        prediction = self.model.predict(x_test)

        color = 'green'
        masks = []
        for i, _mask in enumerate(prediction): 
            mask = mask_to_rgba(
                                zero_pad_mask(prediction[i], desired_size=prediction[i].shape[1]),
                                color=color
                            )
            masks.append(mask)
        masks = np.asarray(masks)

        outs = []
        for i, mask in enumerate(masks):
            out = x_test[i]*0.8 +masks[i]*0.2
            outs.append(out)
        outs = np.asarray(outs)

        width  = int(prediction[i].shape[0])
        height = int(prediction[i].shape[1])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        new_video = cv2.VideoWriter(out_folder_path+'/output_demo.gif', fourcc, 5.0, (width,height))

        for out in outs:
            out = (out*255).astype(np.uint8)
            new_video.write(out)
        new_video.release()



    def load_weights(self, model_path):
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)