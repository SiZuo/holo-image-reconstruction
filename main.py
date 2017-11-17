#from train_4 import *
#from test_4 import *
#from train import *
from train_newImage import *
#from test_newImage import *
import tensorflow as tf
import os

########parameter may need to change: test/train,  batch_size , checkpoint path######

####specify certain GPU to run
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('total_epoch', 80,
                            """Number of epoches to run""")
tf.app.flags.DEFINE_string('data_in_name', 'holo',
                           """Name of dataset to run""")
tf.app.flags.DEFINE_string('data_out_name', 'ground_truth',
                           """Name of dataset to run""")
tf.app.flags.DEFINE_string('data_in_test_name', 'recon',
                           """Name of dataset to run""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of examples in a batch""")
tf.app.flags.DEFINE_float('initial_learning_rates', 0.001,
                          """Initial_learning_rate""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.9,
                          """Parameter for learning rate decay""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 20,
                            """Period of decaying learning rate,100 in previous""")
tf.app.flags.DEFINE_integer('noise total', 0,
                            """Different noise level for training""")
tf.app.flags.DEFINE_integer('train_phase', 1,
                            """Train phase for different network structure""") 
tf.app.flags.DEFINE_integer('range_', 800,
                            """Range of image indexing number, e.g.,(0, 800)""")
tf.app.flags.DEFINE_integer('image_size', 256,
                            """Image size""")
tf.app.flags.DEFINE_integer('image_width', 1280,
                            """Image size""")
tf.app.flags.DEFINE_integer('image_height', 1024,
                            """Image size""")
tf.app.flags.DEFINE_integer('channel_dim', 1,
                            """Color channel dimension""")
tf.app.flags.DEFINE_integer('coef', 0,
                            """coef for regularization""")
tf.app.flags.DEFINE_integer('num_class', 2,
                            """Nuber of classes""")
tf.app.flags.DEFINE_integer('num_gpu', 0,
                            """Number of GPU""")
tf.app.flags.DEFINE_string('phase', 'train',
                           """train or test""")
tf.app.flags.DEFINE_boolean('model_log', False,
                            """Enable/disable log of model""")
tf.app.flags.DEFINE_string('ckpt_name', 'mias2_1',
                           """'dataset_name'+'_'+'batch_size'""")
tf.app.flags.DEFINE_string('ckpt_dir', './checkpoint_without_noise_FusionNet_batch1_BNatBegin_new_newImage',
                           """./checkpoint""")
tf.app.flags.DEFINE_integer('loss_function_change_epoch', 0,
                             """change the loss function with regu to without regu""")

def main(_):
    if FLAGS.phase == 'train':
        if FLAGS.num_gpu == 0:
            train_with_cpu(FLAGS)
        else:
            train_with_gpu(FLAGS)
    else:
        test_with_cpu(FLAGS)


if __name__ == '__main__':
    tf.app.run()
