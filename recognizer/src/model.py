from keras.layers import Conv2D, ReLU, BatchNormalization, Input, GlobalAveragePooling2D, Dense, add, MaxPooling2D
from keras.models import Model
def resnet(input_shape, n_classes):
    def identity_block(y, f):
        x = Conv2D(f, 1)(y)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(f, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(4 * f, 1)(x)
        x = BatchNormalization()(x)
        x = add([x, y])
        x = ReLU()(x)
        return x

    def conv_block(y, f, s):
        x = Conv2D(f, 1)(y)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(f, 3, padding='same', strides=s)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(4 * f, 1)(x)
        x = BatchNormalization()(x)

        z = Conv2D(4 * f, 1,strides=s)(y)
        z = BatchNormalization()(z)

        x = add([x, z])
        x = ReLU()(x)

        return x

    def resnet_block(x, f, s, reps):
        x = conv_block(x, f, s)
        for _ in range(reps - 1):
            x = identity_block(x, f)

        return x

    input = Input(input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = MaxPooling2D(3, strides=2, padding='same')(x)

    x = resnet_block(x, 64, 1, 3)
    x = resnet_block(x, 128, 2, 4)
    x = resnet_block(x, 256, 2, 6)
    x = resnet_block(x, 512, 2, 3)

    x = GlobalAveragePooling2D()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model


if __name__ == '__main__':
    input_shape = (224, 224, 1)  # Updated for grayscale images
    n_classes = 1000
    model = resnet(input_shape, n_classes)
    model.summary()
