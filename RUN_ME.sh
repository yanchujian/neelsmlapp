#!/bin/sh


echo "changing directory"
cd aerial_pedestrian_detection-master

echo "pip install"
pip install . --user

echo "setup.py"
python setup.py build_ext --inplace

echo "beginning training"
python keras_retinanet/bin/train.py  --config config.ini csv train_annotations_new_data.csv labels_new_data.csv

cd /
