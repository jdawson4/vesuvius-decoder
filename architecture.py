# Author: Jacob Dawson

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

# THIS IS CREATING NANs
def get_model(input_shape):
    # I'm going to make my own model, rather than take someone else's.
    # That's kinda the whole fun of this challenge, for me!
    input=keras.Input(input_shape, dtype='float32')
    #output=keras.layers.Rescaling(scale=1/65535.0, offset=0)(input)
    output=keras.layers.Normalization()(input)

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

if __name__=='__main__':
    model = get_model((64,64,64))
    model.summary()
