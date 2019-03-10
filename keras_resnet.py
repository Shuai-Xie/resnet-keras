from keras.layers import Input, Conv2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import MaxPool2D, GlobalAveragePooling2D, Dense
from keras.layers import add  # merge
from keras.models import Model, Sequential
from keras.utils import plot_model

"""implement of Pytorch ResNet"""


def basic_block(input_tensor, filters, stride=1, downsample=None):
    residual = input_tensor
    # 3x3
    x = Conv2D(filters, kernel_size=3, strides=stride, padding='same')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # 3x3
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    # layer 内部没有 fm size 变化，layer连接处有
    if downsample:  # 类比 resnet50，identity_block shortcut 没有 downsample，而 conv_block 有
        residual = downsample(input_tensor)
    x = add([x, residual])
    x = Activation('relu')(x)

    return x


def bottleneck(input_tensor, filters, stride=1, downsample=None):
    residual = input_tensor
    # 1x1
    x = Conv2D(filters, kernel_size=1, padding='same')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # 3x3 core
    x = Conv2D(filters, kernel_size=3, strides=stride, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # 1x1
    x = Conv2D(filters * 4, kernel_size=1, padding='same')(x)  # fms * 4
    x = BatchNormalization(axis=3)(x)
    if downsample:
        residual = downsample(input_tensor)
    x = add([x, residual])
    x = Activation('relu')(x)

    return x


def make_layer(input_tensor, block, filters, blocks, stride=1):
    """
    :param input_tensor:
    :param block: basic_block, bottleneck
    :param filters: 输出通道数
    :param blocks: block 重复次数
    :param stride: stage345, stride=2; stage2, 第一层有maxpool，所以stride=1
    :return: layer output
    """
    global in_filters
    downsample = None
    expansion = 4 if block.__name__ == 'bottleneck' else 1

    """residual 是否需要下采样
      basic_block
        - stage2, stride=1, 并且 in_filters = filters * expansion，所以不用 downsample
      bottleneck
        - stage2，虽然 stride=1，但是 in_filters != filters * expansion，需要 downsample 改变 channel
        - stride != 1，需要把 输入 size/2 再和输出相加，stage3,4,5 每个layer的第1个block
      所以第2个条件 对于不同结构 block 操作不同
    """
    if stride != 1 or in_filters != filters * expansion:
        downsample = Sequential([
            Conv2D(filters * expansion, kernel_size=1, strides=stride, padding='same'),
            BatchNormalization(axis=3)
        ])
    in_filters = filters * expansion  # next layer input filters
    # 对于 stage2，虽然 stride 单独列出，但是也没改变 fm size
    out = block(input_tensor, filters=filters, stride=stride, downsample=downsample)  # layer开始部分
    for i in range(1, blocks):
        out = block(out, filters=filters)  # layer内部重复部分 不需要 downsample，stride=1
    return out


in_filters = 64


def resnet(input_tensor, block, layers, include_top=True, classes=1000):
    # stage1
    x = Conv2D(64, (7, 7), strides=(2, 2),  # 224->112
               padding='same',
               kernel_initializer='he_normal',
               name='conv1')(input_tensor)
    x = BatchNormalization(axis=3, name='conv1_bn')(x)  # channel last
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)  # 112->56

    x = make_layer(x, block, filters=64, blocks=layers[0])  # stage 2
    x = make_layer(x, block, filters=128, blocks=layers[1], stride=2)  # stage 3
    x = make_layer(x, block, filters=256, blocks=layers[2], stride=2)  # stage 4
    x = make_layer(x, block, filters=512, blocks=layers[3], stride=2)  # stage 5

    # fc
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(inputs=[inputs], outputs=[x])
    return model


def build_resnet(input_tensor, name, include_top=True):
    if name == 'resnet18':
        model = resnet(input_tensor, block=basic_block, layers=[2, 2, 2, 2], include_top=include_top)
    elif name == 'resnet34':
        model = resnet(input_tensor, block=basic_block, layers=[3, 4, 6, 3], include_top=include_top)
    elif name == 'resnet50':
        model = resnet(input_tensor, block=bottleneck, layers=[3, 4, 6, 3], include_top=include_top)
    elif name == 'resnet101':
        model = resnet(input_tensor, block=bottleneck, layers=[3, 4, 23, 3], include_top=include_top)
    elif name == 'resnet152':
        model = resnet(input_tensor, block=bottleneck, layers=[3, 8, 36, 3], include_top=include_top)
    else:
        model = None
    return model


inputs = Input(shape=(224, 224, 3))
model = build_resnet(inputs, name='resnet101')
plot_model(model, to_file='resnet101.png', show_shapes=True)
