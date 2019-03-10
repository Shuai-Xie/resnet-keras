from keras.layers import Input, Conv2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import MaxPool2D, GlobalAveragePooling2D, Dense
from keras.layers import add  # merge
from keras.models import Model
from keras.utils import plot_model


def conv_block(input_tensor, kernel_size, filters, stage, block, strides):
    """A block that has a conv layer at shortcut.
        Note:
            kernel_size, only the middle conv use
            stage and block, both used for generating layer names
            strides, used to reduce fm size in first conv layer and shortcut conv layer

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path (core operation)
            filters: list of integers, the filters of 3 conv layer at main path, 3 filters
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides:
                - Strides for the first conv layer in the block.
                - Strides for the shortcut conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3, fm size 减半
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
    assert len(filters) == 3
    filters1, filters2, filters3 = filters

    # conv, bn name
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # bottleneck design, example [64, 64, 256]
    # main path name: a,b,c; shortcut name: 1

    # 1x1 conv, reduce fms 256->64 (last conv_block output 256fms)
    x = Conv2D(filters1, (1, 1), strides=strides,  # stage2, strides=1; stage345, strides=2
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)  # channel last
    x = Activation('relu')(x)

    # 3x3 conv, core operation 128
    x = Conv2D(filters2, kernel_size,  # default strides=1
               padding='same',  # no change fm shape
               kernel_initializer='he_normal',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 1x1 conv, recover fms 128->512
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
    # x = Activation('relu')(x), relu after add

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    # merge, main path 和 shortcut 的 bn 结果 add
    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    与 conv_block 相比，少了参数 strides，没有 shortcut 操作

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        # stage and block, used for generating layer names
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    assert len(filters) == 3
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 1x1, strides use default 1, fm size no change
    x = Conv2D(filters1, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)  # channel last
    x = Activation('relu')(x)

    # 3x3
    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 1x1
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    # shortcut has no conv! cus no need to change fm size
    x = add([x, input_tensor])  # 直接把 input 和 x 相加
    x = Activation('relu')(x)

    return x


def ResNet50(input_tensor, include_top=True, classes=1000):  # 224x224
    # stage1
    # x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)  # 230x230 上下左右
    x = Conv2D(64, (7, 7), strides=(2, 2),  # 224->112
               padding='same',  # 使用 same 可以不用 padding
               kernel_initializer='he_normal',
               name='conv1')(input_tensor)
    x = BatchNormalization(axis=3, name='conv1_bn')(x)  # channel last
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)  # 112->56

    # stage 2 repeat 3 times
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))  # 56x56
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # stage 3 repeat 4 times
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2, 2))  # 28x28
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')  # defa
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # stage 4 repeat 6 times
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=(2, 2))  # 14x14
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # stage 5 repeat 3 times
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(2, 2))  # 7x7
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # fc
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(inputs=[inputs], outputs=[x], name='resnet50')
    return model


# set input
inputs = Input(shape=(224, 224, 3))

# build model
model = ResNet50(inputs)

# plot model structure
plot_model(model, to_file='res50.png', show_shapes=True)
