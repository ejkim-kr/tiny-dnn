

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model

# DNN을 이용한 평균과 합계 모델

TRAIN_ON =0

if TRAIN_ON==1:
 
    x_v =np.random.rand(400,5)
    x_v =x_v *400

    y_v =np.zeros((x_v.shape[0],2))
    y_v[:,0], y_v[:,1] = np.mean(x_v,axis=1), np.sum(x_v,axis=1)

    '''
    #< shuffle
    sf_idx =np.arange(x_v.shape[0])
    np.random.shuffle(sf_idx)
    x_v =x_v[sf_idx,:]; y_v =y_v[sf_idx,:]
    #>
    '''
    SAMPLES =x_v.shape[0]
    TRAIN_SPLIT =int(0.6 * SAMPLES)
    TEST_SPLIT =int(0.2 * SAMPLES + TRAIN_SPLIT)

    x_train, x_test, x_validate = np.split(x_v, [TRAIN_SPLIT, TEST_SPLIT])
    y_train, y_test, y_validate = np.split(y_v, [TRAIN_SPLIT, TEST_SPLIT])

    #> model
    model = tf.keras.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(5,)))
    model.add(layers.Dense(8))
    model.add(layers.Dense(2))
    model.compile(optimizer='Adam', loss='mse', metrics=['mae'])

    history_1 = model.fit(x_train, y_train, epochs=600, batch_size=16, validation_data=(x_validate, y_validate))
    model.save('model_12345.h5')
    #>

    inD =np.array([302., 303., 304., 305., 306.]) #304, 1520
    inD =inD.reshape(-1,5)
    rst =model.predict(inD)
    print(rst)    

else:
    model =load_model('model_12345.h5')
    
    inD =np.random.rand(50,5)
    inD =inD *400

    rst =model.predict(inD)

    # real values
    mcal =np.zeros((50,2))
    mcal[:,0], mcal[:,1] = np.mean(inD,axis=1), np.sum(inD,axis=1)    
    Tmean_acc =0
    Tsum_acc =0
    for i in range(50):
        mean_acc =(mcal[i,0] -abs(rst[i,0] -mcal[i,0])) /mcal[i,0]
        sum_acc = (mcal[i,1] -abs(rst[i,1] -mcal[i,1])) /mcal[i,1]
        Tmean_acc +=mean_acc
        Tsum_acc +=sum_acc
        print(f'{rst[i,0]:08.3f} {rst[i,1]:08.3f} {mcal[i,0]:08.3f} {mcal[i,1]:08.3f}, {mean_acc:05.5f}, {sum_acc:05.5f}')

    print('accuracy=%1.5f'%((Tmean_acc +Tsum_acc) /100))        

