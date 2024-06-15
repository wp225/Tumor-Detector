import numpy as np
import pandas as pd
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self,train_path,test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        train_df = image_dataset_from_directory(
            directory=self.train_path,
            labels='inferred',
            label_mode='categorical',
            batch_size=8,
            class_names=['glioma','meningioma','notumor','pituitary'],
            image_size=(224,224),
            color_mode='grayscale'
        )

        test_df = image_dataset_from_directory(
            directory=self.test_path,
            labels='inferred',
            label_mode='categorical',
            batch_size=8,
            class_names=['glioma','meningioma','notumor','pituitary'],
            image_size= (224,224),
            color_mode='grayscale'
        )

        return train_df,test_df


if __name__ == '__main__':
    image_size = (224,224)
    loader = DataLoader('../../../data/Training','../../../data/Testing')
    train,test= loader.load_data()
    clasname = test.class_names
    for image_batch, labels_batch in train:
        print(image_batch.shape)
        print(labels_batch.shape)
        break
