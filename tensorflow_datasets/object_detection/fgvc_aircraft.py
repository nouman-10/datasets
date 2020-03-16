# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

# Lint as: python3
"""FGVC Aircraft Dataset."""

import os
import re
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_PROJECT_URL = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/'

_DOWNLOAD_URL = ('http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz')

_CITATION = """
@techreport{maji13fine-grained,
   title         = {Fine-Grained Visual Classification of Aircraft},
   author        = {S. Maji and J. Kannala and E. Rahtu
                    and M. Blaschko and A. Vedaldi},
   year          = {2013},
   archivePrefix = {arXiv},
   eprint        = {1306.5151},
   primaryClass  = "cs-cv",
}
"""

_DESCRIPTION = """
The dataset contains 10,200 images of aircraft, with 100 images for each of 102
different aircraft model variants, most of which are airplanes. The (main)
aircraft in each image is annotated with a tight bounding box and a
hierarchical airplane model label.
Aircraft models are organized in a four-levels hierarchy. The four levels, from
finer to coarser, are:
1) Model, e.g. Boeing 737-76J. Since certain models are nearly visually 
indistinguishable, this level is not used in the evaluation.
2) Variant, e.g. Boeing 737-700. A variant collapses all the models that are 
visually indistinguishable into one class. The dataset comprises 102 different
variants.
3) Family, e.g. Boeing 737. The dataset comprises 70 different families.
4) Manufacturer, e.g. Boeing. The dataset comprises 41 different manufacturers.
"""


class FGVCAircraft(tfds.core.GeneratorBasedBuilder):
  """FGVC Aircraft Dataset."""

  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    features = {
        'image':
            tfds.features.Image(encoding_format='jpeg'),
        'image/filename':
            tfds.features.Text(),
        'faces':
            tfds.features.Sequence({
                'bbox': tfds.features.BBoxFeature(),
                'blur': tf.uint8,
                'expression': tf.bool,
                'illumination': tf.bool,
                'occlusion': tf.uint8,
                'pose': tf.bool,
                'invalid': tf.bool,
            }),
    }
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        homepage=_PROJECT_URL,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    extracted_dirs = dl_manager.download_and_extract(_DOWNLOAD_URL)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'split': 'train',
                'extracted_dirs': extracted_dirs
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'split': 'val',
                'extracted_dirs': extracted_dirs
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'split': 'test',
                'extracted_dirs': extracted_dirs
            })
    ]

  def _generate_examples(self, split, extracted_dirs):
    """Yields examples."""
    pattern_fname = re.compile(r'(.*.jpg)\n')
    pattern_annot = re.compile(r'(\d+) (\d+) (\d+) (\d+) (\d+) '
                               r'(\d+) (\d+) (\d+) (\d+) (\d+) \n')
    annot_dir = 'wider_face_split'
    annot_fname = ('wider_face_test_filelist.txt' if split == 'test' else
                   'wider_face_' + split + '_bbx_gt.txt')
    annot_file = os.path.join(annot_dir, annot_fname)
    image_dir = os.path.join(extracted_dirs['wider_' + split], 'WIDER_' + split,
                             'images')
    annot_dir = extracted_dirs['wider_annot']
    annot_path = os.path.join(annot_dir, annot_file)
    with tf.io.gfile.GFile(annot_path, 'r') as f:
      while True:
        # First read the file name.
        line = f.readline()
        match = pattern_fname.match(line)
        if match is None:
          break
        fname = match.group(1)
        image_fullpath = os.path.join(image_dir, fname)
        faces = []
        if split != 'test':
          # Train and val contain also face information.
          with tf.io.gfile.GFile(image_fullpath, 'rb') as fp:
            image = tfds.core.lazy_imports.PIL_Image.open(fp)
            width, height = image.size

          # Read number of bounding boxes.
          nbbox = int(f.readline())
          if nbbox == 0:
            # Cases with 0 bounding boxes, still have one line with all zeros.
            # So we have to read it and discard it.
            f.readline()
          else:
            for _ in range(nbbox):
              line = f.readline()
              match = pattern_annot.match(line)
              if not match:
                raise ValueError('Cannot parse: %s' % image_fullpath)
              (xmin, ymin, wbox, hbox, blur, expression, illumination, invalid,
               occlusion, pose) = map(int, match.groups())
              ymax = np.clip(ymin + hbox, a_min=0, a_max=height)
              xmax = np.clip(xmin + wbox, a_min=0, a_max=width)
              ymin = np.clip(ymin, a_min=0, a_max=height)
              xmin = np.clip(xmin, a_min=0, a_max=width)
              faces.append({
                  'bbox':
                      tfds.features.BBox(
                          ymin=ymin / height,
                          xmin=xmin / width,
                          ymax=ymax / height,
                          xmax=xmax / width),
                  'blur':
                      blur,
                  'expression':
                      expression,
                  'illumination':
                      illumination,
                  'occlusion':
                      occlusion,
                  'pose':
                      pose,
                  'invalid':
                      invalid,
              })
        record = {
            'image': image_fullpath,
            'image/filename': fname,
            'faces': faces
        }
        yield fname, record
