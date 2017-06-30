from keras.layers               import Input, LSTM, RepeatVector
from keras.models               import Model
from keras.layers.core          import Dense
from keras                      import backend as K
from keras.layers.core          import Lambda
from keras.layers.wrappers      import Bidirectional as Bi
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
from keras.optimizers           import Adam, SGD, RMSprop
from keras.layers.core          import Reshape, Flatten
from keras.layers.wrappers      import TimeDistributed as TD

import numpy as np
import sys
import glob
import json
import random

epsilon_std = 0.5
def sampling(args):
  z_mean, z_log_sigma = args
  epsilon = K.random_normal(shape=(batch_size, vae_dim), mean=0., stddev=epsilon_std)
  return z_mean + K.exp(z_log_sigma) * epsilon

batch_size  = 32
timesteps   = 250
input_dim   = 2
latent_dim  = 256
vae_dim     = 16
inputs      = Input(shape=(timesteps, input_dim))
encoded     = LSTM( 128 )(inputs)
z_mean      = Dense( vae_dim )( encoded )
z_log_sigma = Dense( vae_dim )( encoded )
z           = Lambda( sampling, output_shape=( vae_dim ,) )( [z_mean, z_log_sigma] )
encoder     = Model( inputs, encoded )

print( "encoded", encoded.shape )
print( "z", z.shape )
print( "rz", RepeatVector( 4 )( z ) )
decoded     = RepeatVector( timesteps )( z )
decoded     = Bi( LSTM( 128, return_sequences=True) )( decoded )
decoded     = TD( Dense(2, activation='linear') )( decoded )
encoder_decoder = Model(inputs, decoded)

# intermediate_dim = 256
# original_dim     = 1024
# x = Input(batch_shape=(batch_size, original_dim))
# h = Dense(intermediate_dim, activation='relu')(x)


def train():
  xss, yss = [], []
  trains = glob.glob("jsons/train_*.json")
  for o in range(batch_size*200):
    xs = [ [0.0]*2  for i in range(250) ]
    ys = [ [0.0]*2  for i in range(250) ]
    xjson = json.loads( open(trains[o]).read() )
    for index, xreal in enumerate(xjson):
      #print( xreal ) 
      xs[index] = xreal
      ys[index] = xreal
    xss.append( list(reversed(xs)) )
    yss.append( ys )
  xss = np.array( xss )
  yss = np.array( yss )

  loss = 'kullback_leibler_divergence'
  loss = 'mse'
  encoder_decoder.compile(optimizer=Adam(), loss=loss)
  optims  = [ \
              ("Adam", Adam() ), \
              ("RMSprop", RMSprop() ), \
              ("SGD", SGD() ) 
            ]
  for epoch in range(2000):
    name, opt = random.choice( optims )
    encoder_decoder.optimizer = opt
    print("optimizer {name} {optimizer}".format( name = name, optimizer = str(opt) ) )
    encoder_decoder.fit( xss, yss, batch_size=batch_size )
    encoder_decoder.save("models/%09d.h5"%epoch)

def predict():
  tests = glob.glob("jsons/test_*.json")
  xss, yss = [], []
  for o in range(batch_size*1):
    xs = [ [0.0]*2  for i in range(250) ]
    ys = [ [0.0]*2  for i in range(250) ]
    xjson = json.loads( open(tests[o]).read() )
    for index, xreal in enumerate(xjson):
      xs[index] = xreal
      ys[index] = xreal
    xss.append( xs )
    yss.append( ys )
  xss = np.array( xss )
  loss = 'mse'
  encoder_decoder.compile(optimizer=Adam(), loss=loss)
  target = sorted( glob.glob('models/*.h5') ).pop()
  encoder_decoder.load_weights( target ) 
  for xi, xs in enumerate( encoder_decoder.predict( xss ).tolist() ):
    print( xi, xs )
    print( xi, yss[xi] )
  
if __name__ == '__main__':
  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()

