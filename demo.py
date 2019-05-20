# This script shows how to reconstruct from Caffenet features
#
# Alexey Dosovitskiy, 2015

import caffe
import numpy as np
import os
import sys
import patchShow
import scipy.misc
import scipy.io
import cv2

# choose the net
if len(sys.argv) == 3:
    net_name = sys.argv[1]
    iname = sys.argv[2]
else:
    raise Exception('Usage: recon_input.py NET_NAME')

# set up the inputs for the net: 
batch_size = 64
image_size = (3,227,227)
images = np.zeros((batch_size,) + image_size, dtype='float32')

# use crops of the cat image as an example 
in_image = scipy.misc.imread(iname)
print(in_image.shape)
# in_image = scipy.misc.imread('img200.jpg')
#in_image = scipy.misc.imread('img5.jpg')
#in_image = scipy.misc.imresize(img, image_size)
for ni in range(images.shape[0]):
#     images[ni] = np.transpose(in_image[ni:ni+image_size[1], ni:ni+image_size[2]], (2,0,1))
    images[ni] = np.transpose(cv2.resize(in_image, (image_size[1], image_size[2])) , (2,0,1))
# mirror some images to make it a bit more diverse and interesting
images[::2,:] = images[::2,:,:,::-1]
print (images.shape)  
# RGB to BGR, because this is what the net wants as input
data = images[:,::-1] 

# subtract the ImageNet mean
# matfile = scipy.io.loadmat('caffenet/ilsvrc_2012_mean.mat')
# image_mean_1 = matfile['image_mean']
# print(image_mean_1.shape)

image_mean = np.load("/datasets_1/sagarj/BellLabs/Data/cityAugmentedMean.npy")
print(image_mean.shape)


topleft = ((image_mean.shape[0] - image_size[1])/2, (image_mean.shape[1] - image_size[2])/2)
image_mean = image_mean[topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
# del matfile


# data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0) # mean is already BGR
data -= np.expand_dims(image_mean, 0) # mean is already BGR

print("HERE GERE !!", data.shape)
#initialize the caffenet to extract the features
caffe.set_mode_cpu() 
# replace by 
#caffe.set_mode_gpu()
caffenet = caffe.Net('caffenet/caffenet.prototxt', 'caffenet/caffenet.caffemodel', caffe.TEST)
# caffenet = caffe.Net('caffenet/caffenet_deploy_1.prototxt', 'caffenet/caffe_model_beauty_binary_iter_10000.caffemodel', caffe.TEST)

# run caffenet and extract the features
caffenet.forward(data=data)
feat = np.copy(caffenet.blobs[net_name.split('_')[0]].data)
del caffenet

# run the reconstruction net
net = caffe.Net(net_name + '/generator.prototxt', net_name + '/generator_81000.caffemodel', caffe.TEST)
# net = caffe.Net(net_name + '/generator.prototxt', net_name + '/generator_38000_cityAug.caffemodel', caffe.TEST)
# net = caffe.Net(net_name + '/generator.prototxt', net_name + '/dataset_best_58000.caffemodel', caffe.TEST)
generated = net.forward(feat=feat)
topleft = ((generated['generated'].shape[2] - image_size[1])/2, (generated['generated'].shape[3] - image_size[2])/2)
print("THERE THERE !! ", generated['generated'].shape, topleft)
# recon = generated['generated'][:,::-1,topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
recon = generated['generated'][:,::-1,topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
print("Got this!!",recon.shape)
# recon  = cv2.resize(generated['generated'], (image_size[1], image_size[2])) 
# print("Got it!!",recon.shape)
del net

print(images.shape, recon.shape)
collage = np.concatenate((images, recon), axis=3)
print(collage.shape)
# cv2.imwrite("test.png",collage)
# save results to a file
collage = patchShow.patchShow(collage, in_range=(-120,120))
scipy.misc.imsave('reconstructions_' + net_name + '.png', collage)

  
