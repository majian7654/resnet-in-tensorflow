# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
import os
from resnet import *
from datetime import datetime
import time
import pandas as pd
from input_data import read_plantData

IMG_WIDTH = FLAGS.img_width
IMG_HEIGHT = FLAGS.img_height
IMG_DEPTH = 3

class Train(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self):
        # Set up all the placeholders
        self.placeholders()


    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])



    def build_train_validation_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.
        
        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference(self.image_placeholder, reuse=False)
        vali_logits = inference(self.vali_image_placeholder, reuse=True)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss] + regu_losses)

        predictions = tf.nn.softmax(logits)
        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)#calculate the error ratio


        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                    self.train_top1_error)
            self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)



    def train(self):
        '''
        This is the main function for training
        '''

        # Build the graph for train and validation
        self.build_train_validation_graph()

        #read_data_op
        train_batch_data_op, train_batch_labels_op = read_plantData(is_train=True, batch_size=FLAGS.train_batch_size, shuffle=True)

        validation_batch_data_op, validation_batch_labels_op = read_plantData(is_train=False, batch_size=FLAGS.validation_batch_size, shuffle=False)

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config = config)

        #check ckpt
        ckpt = tf.train.get_checkpoint_state('./logs_test_110')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('----------------------------')
            print('Restored from checkpoint...')
            print(ckpt)
            print('----------------------------')
        else:
            sess.run(init)

        # This summary writer object helps write summaries on tensorboard
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)


        # These lists are used to save a csv file at last
        step_list = []
        train_error_list = []
        val_error_list = []

        print('Start training...')
        print('----------------------------')
        
        coord = tf.train.Coordinator();
        threads = tf.train.start_queue_runners(sess, coord = coord)
  
        for step in np.arange(FLAGS.train_steps):

            train_batch_data, train_batch_labels = sess.run([train_batch_data_op, train_batch_labels_op])
            train_batch_labels = train_batch_labels.reshape(-1)

            validation_batch_data, validation_batch_labels = sess.run([validation_batch_data_op, validation_batch_labels_op])
            validation_batch_labels = validation_batch_labels.reshape(-1)
            # Want to validate once before training. You may check the theoretical validation
            # loss first
            if step % FLAGS.report_freq == 0:

                if FLAGS.is_full_validation is True:
                    validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                                            top1_error=self.vali_top1_error, vali_data=vali_data,
                                            vali_labels=vali_labels, session=sess,
                                            batch_data=train_batch_data, batch_label=train_batch_labels)

                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                        simple_value=validation_error_value.astype(np.float))
                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                else:
                    _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                                 self.vali_top1_error,
                                                                                 self.vali_loss],
                                                {self.image_placeholder: train_batch_data,
                                                 self.label_placeholder: train_batch_labels,
                                                 self.vali_image_placeholder: validation_batch_data,
                                                 self.vali_label_placeholder: validation_batch_labels,
                                                 self.lr_placeholder: FLAGS.init_lr})

                val_error_list.append(validation_error_value)


            start_time = time.time()
            _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
                                                                  self.full_loss, self.train_top1_error],
                                {self.image_placeholder: train_batch_data,
                                  self.label_placeholder: train_batch_labels,
                                  self.vali_image_placeholder: validation_batch_data,
                                  self.vali_label_placeholder: validation_batch_labels,
                                  self.lr_placeholder: FLAGS.init_lr})
            duration = time.time() - start_time


            if step % FLAGS.report_freq == 0:
                summary_str = sess.run(summary_op, {self.image_placeholder: train_batch_data,
                                                    self.label_placeholder: train_batch_labels,
                                                    self.vali_image_placeholder: validation_batch_data,
                                                    self.vali_label_placeholder: validation_batch_labels,
                                                    self.lr_placeholder: FLAGS.init_lr})
                summary_writer.add_summary(summary_str, step)

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print(format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                                    sec_per_batch))
                print('Train top1 error = ', train_error_value)
                print('Validation top1 error = %.4f' % validation_error_value)
                print('Validation loss = ', validation_loss_value)
                print('----------------------------')

                step_list.append(step)
                train_error_list.append(train_error_value)



            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print('Learning rate decayed to ', FLAGS.init_lr)

            # Save checkpoints every 10000 steps
            if step % 10000 == 0 or (step + 1) == FLAGS.train_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                'validation_error': val_error_list})
                df.to_csv(train_dir + FLAGS.version + '_error.csv')


    def test(self, test_image_array):
        pass


    ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return: loss tensor with shape [1]
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)

    def train_operation(self, global_step, total_loss, top1_error):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)
        
        #add train images
        tf.summary.image('input',self.image_placeholder)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op


    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op


    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
        Runs validation on all the 10000 valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+FLAGS.validation_batch_size],
                self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)


# Initialize the Train object
train = Train()
# Start the training session
train.train()
