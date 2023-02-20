# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""


#Modify the entire file according to your own training data
import tensorflow as tf
from datasets import pascalvoc_common
import recuop_im_ocua as r

slim = tf.contrib.slim

FILE_PATTERN = 'voc_2007_%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

# (Images, Objects) statistics on every class.
TRAIN_STATISTICS = {
    'none': (0, 0),
    'carrier': (r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[2][0], r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[2][1]),
    'cruiser': (r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[3][0], r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[3][1]),
    'supply':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[9][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[9][1]),
    'tender':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[10][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[10][1]),
    'submarine':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[8][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[8][1]),
    'barge':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[1][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[1][1]),
    'patrol':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[7][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[7][1]),
    'pha':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[11][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[11][1]),
    'fregate':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[5][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[5][1]),
    'destroyer':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[4][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[4][1]),
    'amphiby':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[0][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[0][1]),
    'littoral combat ship':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[6][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','train')[6][1])
    
    
	}
	
	
TEST_STATISTICS = {
   'none': (0, 0),
   'carrier': (r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[2][0], r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[2][1]),
   'cruiser': (r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[3][0], r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[3][1]),
   'supply':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[9][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[9][1]),
   'tender':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[10][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[10][1]),
   'submarine':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[8][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[8][1]),
   'barge':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[1][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[1][1]),
   'patrol':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[7][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[7][1]),
   'pha':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[11][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[11][1]),
   'fregate':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[5][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[5][1]),
   'destroyer':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[4][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[4][1]),
   'amphiby':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[0][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[0][1]),
   'littoral combat ship':(r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[6][0],r.count_Im_occu('VOC2007/ImageSets/test','VOC2007/ImageSets/train','test')[6][1])
	}	

	
	
SPLITS_TO_SIZES = {
    'train': 223,  #Training data volume
    'test': 55,   #Test data volume
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
    'test': TEST_STATISTICS,
}
NUM_CLASSES = 12 #modify according to the actual category of your own data (without background)



def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return pascalvoc_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES)
