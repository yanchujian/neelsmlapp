#!/bin/sh
set -e

gcloud auth activate-service-account neel.r.iyer@gmail.com --key-file=~/key.json
gcsfuse push_bucket /root/output
python app.py --logdir /root/output/experiment