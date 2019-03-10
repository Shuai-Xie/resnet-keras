# resnet-keras

This is an implementation of keras resnet.

You can use ```build_resnet()``` to build different kind of resnet(18,34,50,101,152) by setting name attr.

```py
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
 ```