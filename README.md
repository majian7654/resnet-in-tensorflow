# ResNet in Tensorflow

this repo is an implement of resent!!!

the origin repo is https://github.com/wenxinxu/resnet-in-tensorflow

i modify some part of the repo:

1. modify batch_norm:

   use tf.layer.batch_norm, and this is more convenient for using batchnorm

   in single test

2. add single_demo.py:

   Single_demo.py is for testing a single picture

3. Add multi_demo.py:

   multi_demo.py is for batch test!!!

4. modify resnet.py:

   Make the code is for resnet34

5. modify cifar10_train.py:

   Modify the check if there is checkpoint! if there is , restore and continue training instead of using the hyper_parameters!!! 