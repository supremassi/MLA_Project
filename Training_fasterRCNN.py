########################## Libraries and paths ##########################

import os
import sys
import random
import math   # for mathematical operations

import numpy as np
import pandas as pd

from tqdm import tqdm

import tensorflow as tf

import keras
from keras import backend as K
from keras.preprocessing import image   # for preprocessing the images
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Layer, Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Lambda
from keras.layers import TimeDistributed


dataFolder = str(os.path.abspath('')) + "//Data"    # import the folder 
sys.path.insert(1, dataFolder)

###################### FASTER RCNN ###########################
print("---------- FASTER RCNN -----------")

## PARAMETERS ##
# Anchor box scales
# Original anchor_box_scales in the paper are [128, 256, 512]
anchor_box_scales = [16, 32, 64]
# Anchor box ratios
anchor_box_ratios = [
    [1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
nb_anchors = 9
##


#### VGG16 ####

def modify_model_VGG16(input_size=[224, 224]):
    dim1 = input_size[0]
    dim2 = input_size[1]
    # VGG16 pretrained model from keras without the last FC layers and with the dimension of the input wanted
    model = VGG16(input_shape=(dim1, dim2, 3),
                  include_top=False, weights="vgg16_weights.h5")
    # Create our model without the last layer of max pooling
    my_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    return my_model


#### RPN LAYER ####

def rpn_layer(base_layers, num_anchors):
    y = Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_initializer='normal', name='rpn_conv1')(base_layers)
    y_class = Conv2D(num_anchors, (1, 1), activation='sigmoid',
                     kernel_initializer='uniform', name='rpn_out_class')(y)
    y_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear',
                    kernel_initializer='zero', name='rpn_out_regress')(y)
    return [y_class, y_regr,base_layers]


# Generate anchor boxes

def generate_anc_box(ft_map_img, img, gt_box):
    # shape of the feature map
    ft_map_w = ft_map_img.shape[0]
    ft_map_h = ft_map_img.shape[1]

    # number of the possible anchors and anchor boxes
    anchor_nb = ft_map_w*ft_map_h
    anchor_box_nb = anchor_nb*9

    # x,y intervals to generate anchor box center on the original image
    dx = img.shape[0]//ft_map_w
    dy = img.shape[1]//ft_map_h

    # coordonates of the anchors on the original image
    coordx = np.arange(dx, (ft_map_w+1)*dx, dx)
    coordy = np.arange(dy, (ft_map_h+1)*dy, dy)
    index = 0
    coord = np.zeros((anchor_nb, 2))
    for x in range(len(coordx)):
        for y in range(len(coordy)):
            coord[index, 0] = coordx[x] - dx//2
            coord[index, 1] = coordy[y] - dy//2
            index += 1
    #print("coord shape of anchors of one image", coord.shape)

    # display the anchor_nb anchors points
    # plt.figure(figsize=(9, 6))
    # for i in range(coord.shape[0]):
    #     cv2.circle(img, (int(coord[i][0]), int(coord[i][1])),
    #                radius=1, color=(255, 0, 0), thickness=1)
    # plt.imshow(img)
    # plt.show()

    # for each anchor, generate 9 anchor boxes
    anchor_box_scales = [16, 32, 64]
    # Anchor box ratios
    anchor_box_ratios = [1./np.sqrt(2), 1, np.sqrt(2)]
    anchor_boxes = np.zeros((anchor_box_nb, 4))

    index = 0
    for coord_x, coord_y in coord:  # At each anchor point we generate 9 anchors boxes
        # x,y-coordinates of the current anchor box
        for anchor_size in anchor_box_scales:
            for anchor_ratio in anchor_box_ratios:
                h = anchor_size * (1./anchor_ratio)
                w = anchor_size * anchor_ratio
                anchor_boxes[index, 0] = coord_x - w / 2.
                anchor_boxes[index, 1] = coord_y - h / 2.
                anchor_boxes[index, 2] = coord_x + w / 2.
                anchor_boxes[index, 3] = coord_y + h / 2.
                index += 1

    # print("anchor_boxes shape for one image", anchor_boxes.shape)

    # display the 9 anchor boxes of one anchor and the ground trugh bbox
    # img_clone = np.copy(img)
    # center = ((len(coord)//2)+5)*9
    # for i in range(center, center+9):
    #     x0 = int(anchor_boxes[i][0])
    #     y0 = int(anchor_boxes[i][1])
    #     x1 = int(anchor_boxes[i][2])
    #     y1 = int(anchor_boxes[i][3])
    #     cv2.rectangle(img_clone, (x0, y0), (x1, y1),
    #                   color=(50, 50, 50), thickness=1)
    #     cv2.rectangle(img_clone, (int(gt_box[0]), int(gt_box[1])), (int(
    #         gt_box[2]), int(gt_box[3])), color=(0, 255, 0), thickness=1)
    # plt.imshow(img_clone)
    # plt.show()

    # Ignore cross-boundary anchor boxes : (x1,y1,x2,y2)
    # valid anchor boxes with (x1, y1)>0 and (x2, y2)<=image size
    index_inside = np.where(
        (anchor_boxes[:, 0] >= 0) &
        (anchor_boxes[:, 1] >= 0) &
        (anchor_boxes[:, 2] <= img.shape[0]) &
        (anchor_boxes[:, 3] <= img.shape[1])
    )[0]

    #print("valid index shape", index_inside.shape)

    valid_anchor_boxes = anchor_boxes[index_inside]

    # Display all anchor boxes
    # img_clone = np.copy(img)
    # for i in range(len(anchor_boxes)):
    #     x0 = int(anchor_boxes[i][0])
    #     y0 = int(anchor_boxes[i][1])
    #     x1 = int(anchor_boxes[i][2])
    #     y1 = int(anchor_boxes[i][3])
    #     cv2.rectangle(img_clone, (x0, y0), (x1, y1), color=(255, 255, 2550), thickness=1)
    # plt.imshow(img_clone)
    # plt.show()

    #print("valid_anchor_boxes shape", valid_anchor_boxes.shape)
    return valid_anchor_boxes


def transf_anch_box(anc_box):
    x1, y1, x2, y2 = anc_box[0], anc_box[1], anc_box[2], anc_box[3]
    w = x2-x1
    h = y2-y1
    return np.array([x1, y1, w, h])

# IOU TO GENERATE POS AND NEG ANCHOR BOXES


def union(au, bu, area_intersection):
    # (x2-x1)*(y2-y1)
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u)


def labelise_anc_box(anc, gt_box):
    anc_boxes = []
    anc_labels = []
    for anc_box in anc:
        iou_anc = iou(anc_box, gt_box)
        if iou_anc >= 0.7:
            anc_boxes.append(anc_box)
            anc_labels.append(1)
        elif iou_anc <= 0.3:
            anc_boxes.append(anc_box)
            anc_labels.append(0)
        else:
            anc_boxes.append(anc)
            anc_labels.append(-1)
    return anc_boxes, anc_labels


def batch_valid_anc(anc_labels):
    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256
    pos_idx = np.where(anc_labels == 1)
    neg_idx = np.where(anc_labels == 0)

    if len(pos_idx) > num_regions/2:
        non_pos_idx = random.sample(
            range(len(pos_idx)), len(pos_idx) - num_regions//2)
        # we desactivate the n_pos-128 anchors to have only 128 positive anchors
        anc_labels[non_pos_idx] = -1

    if len(neg_idx) > num_regions/2:
        non_neg_idx = random.sample(
            range(len(neg_idx)), len(neg_idx) - num_regions//2)
        anc_labels[non_neg_idx] = -1

    return anc_labels


# CLASSIFICATION

# ROI POOLING
class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''

    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_data_format()
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            rs = tf.image.resize(
                img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 3)
        final_output = K.reshape(
            final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# CLASSIFICATION


def classification(ft_map, input_rois, num_rois, nb_classes):

    pooling_regions = 7

    # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # num_rois (4) 7x7 roi pooling
    out_roi_pool = RoiPoolingConv(
        pooling_regions, num_rois)([ft_map, input_rois])

    # 2 FC and 2 dropout
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    # object classification
    Y_class = TimeDistributed(Dense(nb_classes, activation='softmax',
                                    kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # bounding boxes regression
    Y_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear',
                                   kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [Y_class, Y_regr]


# Loss Functions

def rpn_loss_cls():
    """Loss function for rpn classification
    """
    def rpn_loss_cls_fixed_num(y_true_cls, y_pred_cls):
        loss = keras.losses.BinaryCrossentropy(from_logits=True)
        return loss(y_true_cls, y_pred_cls).numpy()

    return rpn_loss_cls_fixed_num


def rpn_loss_regr():
    # Loss function for rpn regretion cross entropy between y_pred_regr from rpn and y_true_regr from our generate's anchor boxes

    def rpn_loss_regr_fixed_num(y_true_regr, y_pred_regr):
        loss = keras.losses.CategoricalCrossentropy()
        return loss(y_true_regr, y_pred_regr).numpy()

    return rpn_loss_regr_fixed_num


def classification_loss_cls(y_true_cls, y_pred_cls):
    loss = keras.losses.CategoricalCrossentropy()
    return loss(y_true_cls, y_pred_cls).numpy()


def classification_loss_regr():
    def rpn_loss_regr_fixed_num(y_true_regr, y_pred_regr):
        loss = keras.losses.CategoricalCrossentropy()
        return loss(y_true_regr, y_pred_regr).numpy()
    return rpn_loss_regr_fixed_num


###################### DATA PROCESS ###########################
print("---------- DATA PROCESSING -----------")

im_size = 224

# We collect data from our csv file
# = str(Path(__file__).parent)  # Folder of this file --> "Project"
filePath = str(os.path.abspath(''))
csv_filename = "//Data//RGB//UCF101//data_RGB_train.csv"
name_file = filePath+"//"+csv_filename

train_data = pd.read_csv(name_file, sep=";")

# creating an empty list
train_image = []

# for loop to read and store frames
for i in tqdm(range(1)):
    # loading the image and keeping the target size as (224,224,3)
    filePath = os.path.abspath('')
    path = filePath+"//Data//RGB//UCF101//TRAIN//" + \
        train_data["file_name"].values.tolist()[i]  # Image File
    orig_img = image.load_img(path)
    orig_img_size = orig_img.size[0:2]
    img = image.load_img(path, target_size=(im_size, im_size, 3))
    img_size = img.size[0:2]
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img/255
    # appending the image to the train_image list
    train_image.append(img)

# converting the list to numpy array
X_train = np.array(train_image)
# shape of the array
print("X_train shape : ", X_train.shape)

# GT labels for train and validation set
y_train = np.array(train_data["label"].tolist()[:1])
print("y_train shape : ", y_train.shape)
# y_test = pd.get_dummies(y_test)

# creating list of the ground truth's bounding boxes
coord = ['x1', 'y1', 'x2', 'y2']
gt_boxes = train_data[coord]
gt_boxes = np.array(gt_boxes.values.tolist()[:1])*([img_size[0]/orig_img_size[0],
                                                    img_size[0]/orig_img_size[1], img_size[0]/orig_img_size[0], img_size[0]/orig_img_size[1]])
print("gt_boxes shape : ", gt_boxes.shape)


###################### TRAIN MODEL ###########################
print("---------- TRAIN MODEL -----------")

# VGG
base_model = modify_model_VGG16(input_size=[224, 224])
ft_maps = base_model.predict(X_train)
print("Feature maps shape: ", ft_maps.shape)
print(ft_maps[:, :, 0].shape)
print("Image shape ", X_train[0].shape)

srt = np.array(ft_maps[0])
srt = srt[:, :, 500]*255
# visualize a channel of the feature maps
# plt.imshow(srt2, cmap='gray')
# plt.show()

# RPN
anc_boxes = []
anc_labels = []
for i, img in enumerate(X_train):
    # generate anchor boxes with the image and the feature map  1256
    anc = generate_anc_box(ft_maps[i], img, gt_boxes[i])
    # labelise positive and negative anchor boxes with the IoU with the gt boxes
    anc, lab = labelise_anc_box(anc, gt_boxes[i])
    anc_boxes.append(anc)
    # we take batch of 128 pos and 128 neg anchor boxes
    anc_labels.append(batch_valid_anc(lab))
anc_boxes = np.array(anc_boxes, dtype=object)
anc_labels = np.array(anc_labels, dtype=object)
print("Anchor boxes : ", np.shape(anc_boxes))

img_input = Input(shape=(None, None, 3))
num_anchors = len(anchor_box_ratios)*len(anchor_box_scales)
rpn = rpn_layer(ft_maps, num_anchors)
scale = tf.Variable(1.) 
model_rpn = Model(img_input, Lambda(lambda x: x * scale)(rpn[:2]))

# classifier
nb_classes = 24  # FOR UCF101
roi_input = Input(shape=(None, 4))
classifier = classification(ft_maps, roi_input, 4, nb_classes)

model_classifier = Model([img_input, roi_input], Lambda(lambda x: x * scale)(classifier))

model = Model([img_input, roi_input], Lambda(lambda x: x * scale)(rpn[:2] + classifier))

# TRAINING
print("--------TRAINING--------")
optimizer = keras.optimizers.SGD(learning_rate=0.001)
optimizer_classifier = keras.optimizers.SGD(learning_rate=0.001)
model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(), rpn_loss_regr()])
model_classifier.compile(optimizer=optimizer_classifier, loss=[classification_loss_cls, classification_loss_regr(
)], metrics={'dense_class_{}'.format(len(nb_classes)): 'accuracy'})
model.compile(optimizer='sgd', loss='mae')

Y_pred = model.predict(X_train)

print(Y_pred)
