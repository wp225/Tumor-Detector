import keras.losses

from recognizer.src.utils.data_loader import DataLoader
from recognizer.src.model import resnet
import logging
import tensorflow as tf
import os

# Set environment variable to force Metal backend
os.environ['TF_METAL_ENABLE'] = '1'

class  ModelTrainer:
    def __init__(self,input_shape,n_classes,train_path,test_path):
        print("TensorFlow GPU Support:", tf.config.list_physical_devices('GPU'))

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.train_path = train_path
        self.test_path = test_path

    def train(self):
        model = resnet(self.input_shape,self.n_classes)
        model.summary()

        loader = DataLoader(self.train_path,self.test_path)
        train_df, test_df = loader.load_data()
        print(f'Train Classes: {train_df.class_names}')
        print(f'Test Classes: {train_df.class_names}')

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_df.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = test_df.cache().prefetch(buffer_size=AUTOTUNE)

        model.compile(
            optimizer='adam',
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        history = model.fit(
            train_df,
            validation_data=test_df,
            epochs=10,  # Adjust epochs as needed
        )



if __name__ == '__main__':
    input_shape = (224,224,1)
    n_classes = 4
    train_path = './data/Training'
    testing_path = './data/Testing'

    trainer = ModelTrainer(input_shape,n_classes,train_path,testing_path)
    trainer.train()

