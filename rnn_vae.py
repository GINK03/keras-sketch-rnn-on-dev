from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from keras.layers.core import Dense
from keras import backend as K
from keras.layers.core import Lambda

epsilon_std = 0.5
def sampling(args):
  z_mean, z_log_sigma = args
  epsilon = K.random_normal(shape=(batch_size, latent_dim),
                            mean=0., stddev=epsilon_std)
  return z_mean + K.exp(z_log_sigma) * epsilon

batch_size  = 32
timesteps   = 50
input_dim   = 128
latent_dim  = 256
inputs      = Input(shape=(timesteps, input_dim))
encoded     = LSTM(latent_dim)(inputs)
z_mean      = Dense( latent_dim )( encoded )
z_log_sigma = Dense( latent_dim )( encoded )
z           = Lambda( sampling, output_shape=(latent_dim,) )( [z_mean, z_log_sigma] )
encoder     = Model( inputs, z )

decoded     = RepeatVector( timesteps )( z )
decoded     = LSTM(input_dim, return_sequences=True)( decoded )

encoder_decoder = Model(inputs, decoded)

# intermediate_dim = 256
# original_dim     = 1024
# x = Input(batch_shape=(batch_size, original_dim))
# h = Dense(intermediate_dim, activation='relu')(x)



