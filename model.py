from keras_unet.models import custom_unet
from tensorflow.keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded

class CustomUnetMpdel():
    """Unet model for Lumb Segmentation
    
    Attributes
    ----------
    input_shape : int
        3D Tensor of shape (x, y, num_channels)
    """
    def __init__(self, input_shape):
        model = custom_unet(
            input_shape,
            filters=32,
            use_batch_norm=True,
            dropout=0.3,
            dropout_change_per_layer=0.0,
            num_layers=4
        )
        model.summary()
        model.compile(
            optimizer=Adam(), 
            loss='binary_crossentropy',
            metrics=[iou, iou_thresholded]
        )
        self.model = model

    def train(self, dataset_path = 'data/ratings_train.dat'):
        from utils import get_augmented_data_loader
        train_gen = get_augmented_data_loader('data/ratings_train.dat')

        model.fit_generator(
            train_gen,
            steps_per_epoch=200,
            epochs=8
        )

    def evaluate(self, dataset_path = 'data/ratings_train.dat'):
        pass

    def load_weights(self):
        pass