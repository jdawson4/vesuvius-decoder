# Author: Jacob Dawson
#
# important note: the original model is ~2 million params!
# it also appears to be using an addition-based residual u-net. I wonder if
# this can be done better using concatenation/densenets, either with conv
# layers or perhaps with self-attention? This is where I want to experiment!

from tensorflow import keras

def convBlock(x1, filters, apply_batchnorm=False, apply_dropout=False):
    x2=keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding="same",activation="selu")(x1)
    c2=keras.layers.Concatenate()([x1,x2])
    out=keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding="same",activation="selu")(c2)
    #c3=keras.layers.Concatenate()([x1,x2,x3])
    #x4=keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding="same",activation="selu")(c3)
    #c4=keras.layers.Concatenate()([x1,x2,x3,x4])
    #x5=keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding="same",activation="selu")(c4)

    if apply_batchnorm:
        out = keras.layers.BatchNormalization()(out)
    if apply_dropout:
        out = keras.layers.Dropout(0.25)(out)

    return out

def downsample(x, filters, apply_batchnorm=False, apply_dropout=False):
    out=keras.layers.Conv2D(filters=filters,kernel_size=4,strides=2,padding="same",activation="selu")(x)

    if apply_batchnorm:
        out = keras.layers.BatchNormalization()(out)
    if apply_dropout:
        out = keras.layers.Dropout(0.25)(out)

    return out

def upsample(x, filters, apply_batchnorm=False, apply_dropout=False):
    out=keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(x)
    out=keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding="same",activation="selu")(out)

    if apply_batchnorm:
        out = keras.layers.BatchNormalization()(out)
    if apply_dropout:
        out = keras.layers.Dropout(0.25)(out)

    return out

# this has exactly 3,922,177 trainable params
def convModel(input_shape):
    # I'm going to make my own model, rather than take someone else's.
    # That's kinda the whole fun of this challenge, for me!
    # this model is based on convolutional networks I've made before. This one
    # is a U-net with dense layers and connections "across" the U. Lots of
    # connections everywhere, maybe make a graph of this to visualize!
    input=keras.Input(input_shape, dtype='float32')
    output=keras.layers.Rescaling(scale=1./127.5, offset=-1.)(input) # MAKE THIS CHOICE BASED ON MIN/MAX FROM THE MAIN SCRIPT!

    output1024=convBlock(output,64,False,True) # output size 1024
    output512=downsample(output1024,64,False,True) # output size 512

    output512=convBlock(output512,64,True,False) # output size 512
    output256=downsample(output512,64,True,False) # output size 256

    output256=convBlock(output256,128,True,False) # output size 256
    output128=downsample(output256,256,True,False) # output size 128

    output128=convBlock(output128,256,True,False) # output size 128
    output256=keras.layers.Concatenate()([output256,upsample(output128,128,True,False)]) # output size 256

    output256=convBlock(output256,64,True,False) # output size 256
    output512=keras.layers.Concatenate()([output512,upsample(output256,64,True,False)]) # output size 512

    output512=convBlock(output512,64,True,False)
    output1024=keras.layers.Concatenate()([output1024,upsample(output512,64,True,False)]) # output size 1024, original size!

    output=keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding="same",activation="selu")(output1024)
    output=keras.layers.Conv2D(filters=32,kernel_size=3,strides=1,padding="same",activation="selu")(output)
    output=keras.layers.Conv2D(filters=1,kernel_size=1,strides=1,padding="same",activation="sigmoid")(output)

    model = keras.Model(input, output)
    return model

def attnBlock(input, heads, keyDim, outShape, apply_batchnorm=False, apply_dropout=False):
    out=keras.layers.MultiHeadAttention(heads,keyDim,output_shape=outShape)(input,input)

    out = keras.layers.Concatenate()([input, out])

    if apply_batchnorm:
        out=keras.layers.BatchNormalization()(out)

    out=keras.layers.Conv2D(filters=outShape,kernel_size=1,strides=1,padding='same')(out)
    out = keras.layers.Activation("selu")(out)

    if apply_dropout:
        out = keras.layers.Dropout(0.25)(out)

    return out

# this has exactly 3,834,705 trainable params
def attentionModel(input_shape):
    # this model will be based on attention rather than convolutions.
    # Let's see which gives better performance!
    input=keras.Input(input_shape, dtype='float32')
    output=keras.layers.Rescaling(scale=1./127.5, offset=-1.)(input) # MAKE THIS CHOICE BASED ON MIN/MAX FROM THE MAIN SCRIPT!

    # we'll actually do the same thing as above--we'll make a U-net
    # architecture with convolutional up- and down-scaling, but instead of the
    # dense convblocks, we'll use attention-based resonant blocks!

    output1=attnBlock(output,8,32,64,False,True)
    output2=downsample(output1,128,False,True)

    output2=attnBlock(output2,8,32,64,True,False)
    #output3=downsample(output2,128,True,False)

    #output3=attnBlock(output3,8,32,128,True,False)
    #output4=downsample(output3,512,True,False)

    #output4=attnBlock(output4,8,32,256,True,False)
    #output3=keras.layers.Concatenate()([output3,upsample(output4,256,True,False)])

    #output3=attnBlock(output3,8,32,128,True,False)
    #output2=keras.layers.Concatenate()([output2,upsample(output3,128,True,False)])

    output2=attnBlock(output2,8,32,64,True,False)
    output1=keras.layers.Concatenate()([output1,upsample(output2,64,True,False)])

    output=attnBlock(output1,8,32,32,True,False)
    output=attnBlock(output,8,32,16,True,False)
    output=keras.layers.Conv2D(filters=1,kernel_size=1,strides=1,padding="same",activation="sigmoid")(output)

    model = keras.Model(input, output)
    return model

if __name__=='__main__':
    #model = convModel((64,64,64))
    model = attentionModel((64,64,64))
    model.summary()
