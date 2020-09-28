from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D
from tensorflow.keras.layers import concatenate, add, GlobalMaxPool3D
from tensorflow.keras.activations import relu


def conv_BN_block(input_arr, num_filters, kernel_size, batch_norm):
    
    x = Conv3D(num_filters,
               kernel_size=(kernel_size, kernel_size, kernel_size),
               strides=(1,1,1),
               padding='same')(input_arr)
    
    if batch_norm:
        x = BatchNormalization()(x)
        
    x = Activation('relu')(x)
    
    x = Conv3D(num_filters,
               kernel_size=(kernel_size, kernel_size, kernel_size),
               strides=(1,1,1),
               padding='same')(x)
    
    if batch_norm:
        x = BatchNormalization()(x)
        
    x = Activation('relu')(x)
    
    return x


def Unet_3D(input_arr, n_filters = 8, dropout = 0.2, batch_norm = True):
    
    ### down-sampling
    conv1 = conv_BN_block(input_arr=input_arr, num_filters=n_filters, kernel_size=3, batch_norm=batch_norm)
    mp1   = MaxPooling3D(pool_size=(2,2,2), strides=2)(conv1)
    dp1   = Dropout(dropout)(mp1)
  
    conv2 = conv_BN_block(dp1, n_filters*2, 3, batch_norm)
    mp2   = MaxPooling3D(pool_size=(2,2,2), strides=2)(conv2)
    dp2   = Dropout(dropout)(mp2)

    conv3 = conv_BN_block(dp2, n_filters*4, 3, batch_norm)
    mp3   = MaxPooling3D(pool_size=(2,2,2), strides=2)(conv3)
    dp3   = Dropout(dropout)(mp3)
  
    conv4 = conv_BN_block(dp3, n_filters*8, 3, batch_norm)
    mp4   = MaxPooling3D(pool_size=(2,2,2), strides=2)(conv4)
    dp4   = Dropout(dropout)(mp4)
  
    conv5 = conv_BN_block(dp4, n_filters*16, 3, batch_norm)
    
    conv6 = Conv3D(n_filters*16, kernel_size=(2,2,2), strides =(2,2,2) , padding='same')(conv5)
    dp6   = Dropout(dropout)(conv6)

    conv7 = conv_BN_block(dp6, n_filters*16, 5, True)
    
    
    ### up-sampling
    up1   = Conv3DTranspose(n_filters*8, (2,2,2), strides=(2,2,2), padding='same')(conv7)
    conc1 = concatenate([up1, conv5]);
    conv8 = conv_BN_block(conc1, n_filters*16, 5, True)
    dp7   = Dropout(dropout)(conv8)
    
    up2   = Conv3DTranspose(n_filters*4, (2,2,2), strides=(2,2,2), padding='same')(dp7)
    conc2 = concatenate([up2, conv4])
    conv9 = conv_BN_block(conc2, n_filters*8, 5, True)
    dp8   = Dropout(dropout)(conv9)
    
    up3   = Conv3DTranspose(n_filters*2, (2,2,2), strides=(2,2,2), padding='same')(dp8)
    conc3 = concatenate([up3, conv3])
    conv10= conv_BN_block(conc3, n_filters*4, 5, True)
    dp9   = Dropout(dropout)(conv10)
    
    up4   = Conv3DTranspose(n_filters,   (2,2,2), strides=(2,2,2), padding='same')(dp9)
    conc4 = concatenate([up4, conv2])
    conv11= conv_BN_block(conc4, n_filters*4, 5, True)
    dp10  = Dropout(dropout)(conv11)
    
    up5   = Conv3DTranspose(n_filters,   (2,2,2), strides=(2,2,2), padding='same')(dp10)
    conc5 = concatenate([up5, conv1])
    conv12= Conv3D(n_filters*2,kernel_size=(5,5,5), strides=(1,1,1), padding='same')(conc5)
    dp11  = Dropout(dropout)(conv12)
    
    add1  = add([dp11, conc5])
    relu_layer = relu(add1, max_value=1)
    outputs = Conv3D(1, (1,1,1), activation='relu')(relu_layer)

    model = Model(inputs=input_arr, outputs=outputs)

    return model
