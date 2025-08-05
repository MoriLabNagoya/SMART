from keras.models import *
from keras.layers import *

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder

def se_block(input_tensor, reduction_ratio=4):
    """
    Squeeze-and-Excitation Block
    Args:
        input_tensor: Input feature map
        reduction_ratio: Reduction ratio for bottleneck
    Returns:
        Output tensor after applying SE block
    """
    channels = input_tensor.shape[-1]  # 获取通道数
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)
    se = Dense(channels // reduction_ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    return Multiply()([input_tensor, se])
def segnet_decoder(f, n_classes, n_up=3):

    assert n_up >= 2

    o = f
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = se_block(o)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    o = se_block(o)
    for _ in range(n_up-2):
        o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid',
             data_format=IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = se_block(o)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING, name="seg_feats"))(o)
    o = (BatchNormalization())(o)
    o = se_block(o)
    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    return o


def _segnet(n_classes, encoder,  input_height=416, input_width=608,
            encoder_level=3, channels=3):

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width, channels=channels)

    feat = levels[encoder_level]
    o = segnet_decoder(feat, n_classes, n_up=3)
    o = UpSampling2D(size=(2,2))(o)
    model = get_segmentation_model(img_input, o)

    return model


def segnet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):

    model = _segnet(n_classes, vanilla_encoder,  input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level, channels=channels)
    model.model_name = "segnet"
    return model


def vgg_segnet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):

    model = _segnet(n_classes, get_vgg_encoder,  input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level, channels=channels)
    model.model_name = "vgg_segnet"
    return model


def resnet50_segnet(n_classes, input_height=416, input_width=608,
                    encoder_level=3, channels=3):

    model = _segnet(n_classes, get_resnet50_encoder, input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level, channels=channels)
    model.model_name = "resnet50_segnet"
    return model


def mobilenet_segnet(n_classes, input_height=224, input_width=224,
                     encoder_level=3, channels=3):

    model = _segnet(n_classes, get_mobilenet_encoder,
                    input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level, channels=channels)
    model.model_name = "mobilenet_segnet"
    return model


if __name__ == '__main__':
    m = vgg_segnet(101)
    m = segnet(101)
    # m = mobilenet_segnet( 101 )
    # from keras.utils import plot_model
    # plot_model( m , show_shapes=True , to_file='model.png')
