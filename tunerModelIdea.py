import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Dropout, AveragePooling1D, SpatialDropout1D, GlobalMaxPooling1D
from tensorflow.keras import Model
import keras_tuner as kt

def build_model(hp):
    dropout = hp.Float('dropout', .02, .5, step=.02)
    sDropout = hp.Float('sparseDropout', 0, .14, step=.02)
    activation = 'elu'
    #activation = hp.Choice('activaion', values=['elu', 'sigmoid', 'selu'])
    #convActivation = hp.Choice('convActivation', values=['elu', 'sigmoid', 'exponential', 'selu'])

    #l2 = hp.Choice('l2', values=[1e-4, 1e-5, 1e-6, 0.0])
    l2 = hp.Float('l2', 1e-8, 1e-4, sampling = 'log')

    regularizer = None
    if l2 > 0:
        regularizer = keras.regularizers.l2(l2)

    #Defining dense layer starting hyperparameters
    denseUnits = hp.Int('dense', 12, 90, step=6)
    denseMult = hp.Float('denseMult', .25, .75, step=.1)

    #Fill list with first layer's units
    dense_layers = [denseUnits]
    
    #Define units for 3 more dense layers
    nextUnits = denseUnits
    for layer in range(1, 4):
        nextUnits = int(nextUnits * denseMult)

        if (nextUnits >= 5):
            dense_layers.append(nextUnits)

    #learningRate = hp.Choice('learningRate', values=[1e-2, 1e-3, 1e-4], ordered = True)
    learningRate = 1e-4
    
    timeSteps = 1500
    reduction = hp.Int('reduction', 1, 4, step = 1, default = 1) 
    timeSteps = timeSteps / reduction

    kernelSize = 7
    poolSize = 2
    poolStride = 2

    #Defining convolutional layer starting hyperparameters
    convLayers = hp.Int('convLayers', 4, 10, step=2)
    startFilters = hp.Int('startFilters', 25, 100, step=25)
    convMult = hp.Float('convMult', 1.1, 1.4, step=.1)

    #Fill list with first layer's filters
    conv_layers = [startFilters]
    timeSteps = round((timeSteps - (kernelSize - 1)) / poolStride)

    #Define filters for remaining conv layers
    nextFilters = startFilters
    for layer in range(1, convLayers):
        if (timeSteps - (kernelSize - 1)) / poolStride >= 1:
            timeSteps = int(round((timeSteps - (kernelSize - 1)) / poolStride))
            nextFilters = int(nextFilters * convMult)
            conv_layers.append(nextFilters)

    #I'm just going to hardcode these because I don't really know how to pass them in
    nclasses=5
    timeSteps = 1500
    channels = 51

    model = create_model(timeSteps, channels, nclasses, conv_layers, kernelSize, poolSize, poolStride, reduction, dense_layers, dropout, sDropout, regularizer, activation, learningRate)
    return model

def create_model(ntimeSteps, nchannels, nclasses, convLayers, kernelSize, poolSize, poolStride, reduction, denseLayers, dropout, sDropout, regularizer, activation, learningRate):
    inputTensor = Input(shape = (ntimeSteps, nchannels), name = 'input')

    previousTensor = inputTensor #previousTensor is used to link all of the tensors together

    #Used to reduce the timesteps immediately
    previousTensor=AveragePooling1D(
                pool_size=reduction, 
                strides = reduction,
                name='AdvPool',
                padding = 'same'
            )(previousTensor)
    
    
    count=0 #used to name layers
    #Creates the convolution layers
    for i in convLayers:
        name = 'C%02d'%count
        previousTensor = Convolution1D(
            filters = i,
            kernel_size = kernelSize,
            strides = 1,
            padding = 'valid',
            use_bias = True,
            kernel_initializer = 'random_uniform',
            bias_initializer = 'zeros',
            name = name,
            activation = activation,
            kernel_regularizer=regularizer
        )(previousTensor)

        if(poolStride > 1):
            name='Max%02d'%count
            previousTensor=MaxPooling1D(
                pool_size= poolSize, 
                strides = poolStride,
                padding = 'same',
                name=name
            )(previousTensor)
        
        if sDropout > 0:
            name='SpatialDrop%.02d'%count
            previousTensor=SpatialDropout1D(sDropout, name=name)(previousTensor)
            
        count+=1

    previousTensor = GlobalMaxPooling1D()(previousTensor)

    count = 0#used to name layers
    #Creates the dense layers
    for i in denseLayers:
        name = 'D%02d'%count
        previousTensor = Dense(
            units = i,
            activation = activation,
            use_bias = 'True',
            bias_initializer = 'zeros',
            name = name,
            kernel_regularizer=regularizer
        )(previousTensor)

        if dropout > 0:
            name='Drop%02d'%count
            previousTensor=Dropout(dropout, name=name)(previousTensor)
        count+=1

    output = Dense(
        units = nclasses,
        activation = 'softmax',
        bias_initializer = 'zeros',
        name = 'output'
    )(previousTensor)

    model = Model(inputs = inputTensor, outputs = output) #Actually creates the model
    opt = tf.keras.optimizers.Adam(learning_rate=learningRate, beta_1=.9, beta_2=.999, epsilon=None, decay=0.0, amsgrad=False) #Creates the optimizer
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['categorical_accuracy']) #Compiles the model with the optimizer and metrics
    print(model.summary())
    return model
