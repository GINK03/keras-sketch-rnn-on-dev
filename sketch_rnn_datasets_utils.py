from __future__ import print_function
import numpy as np
import sys
import os
import json

def check_structure():
  """
  please open npz in Python 2, if you would open npz in Python3, UnicoderError occures
  """
  data = np.load("sketch-rnn-datasets/aaron_sheep/aaron_sheep.npz") 
  maxs = []
  mins = []
  lens = []
  for key, values in  data.items() :
    for e, value in  enumerate( values.tolist() ):
      params  = map(lambda x:x.tolist(), value)
      flatten = sum(params, [])
      maxx    = max(flatten)
      minn    = min(flatten)
      maxs.append( maxx )
      mins.append( minn )
      lens.append( len(params) )
  minmax = {
            'max_average' :  sum(maxs)/len(maxs), 
            'min_average' :  sum(mins)/len(mins) 
           }
  print( minmax, max(lens) )
  open('minmax.json', 'w').write( json.dumps(minmax) )
  """
  example...
  {'max_average': 100, 'min_average': -123} 250
  """
  def shrink(xs):
    xs       = xs.tolist()
    delta_x  = xs[0]/100.0 # -> to minimize range ( -1 to +1  )
    delta_y  = xs[1]/100.0
    pen_left = xs[2]
    return [delta_x, delta_y]
  for key, values in  data.items() :
    for e, value in  enumerate( values.tolist() ):
      params  = map(lambda x:shrink(x), value)
      name    =  "%s_%09d.json"%(key, e)
      open("jsons/%s"%name, 'w').write( json.dumps(params) )
      #print( name , params ) 



if __name__ == '__main__':
  if '--step1' in sys.argv:
     check_structure()
