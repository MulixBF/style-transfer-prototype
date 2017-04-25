import numpy as np
import sys
import os
import scipy.misc
import tensorflow as tf
import PIL

sys.path.append('fast-style-transfer/src/')
import transform, utils


class Prisma:
    
    STYLE_DIR = 'styles/'
    STYLES = ['rain_princess', 'scream', 'udine', 'wave', 'wreck']
    
    def __init__(self, image_resolution=(300, 250)):

        self._image_resolution = image_resolution 
        
        self._graph=tf.Graph()
        self._session = tf.Session(graph=self._graph)
        
        with self._graph.as_default():
            self._image_placeholder = tf.placeholder(dtype=tf.float32, 
                                                 shape=(1, 300, 250, 3), 
                                                 name='img_placeholder')
            
            self._output = transform.net(self._image_placeholder)

    def process_image(self, image, style):
               
        with tf.Session(graph=self._graph) as session:
            
            resized_image= scipy.misc.imresize(image, (self._image_resolution))
        
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(self.STYLE_DIR, f'{style}.ckpt'))        
            result = session.run(self._output, feed_dict={self._image_placeholder: [resized_image]})
            result = np.clip(result[0], 0, 255).astype(np.uint8)
            return PIL.Image.fromarray(result)
