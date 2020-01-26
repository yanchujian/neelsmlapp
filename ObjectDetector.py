import cv2 as cv
import numpy
import keras

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import os
import time
import tensorflow as tf

import argparse
import time

from PIL import Image
from PIL import ImageDraw

import io
#import detect
#import tflite_runtime.interpreter as tflite
from tensorflow.compat.v1.lite import Interpreter
#from tensorflow.compat.v2.lite import Interpreter
#from tensorflow.contrib.lite import Interpreter

# additional shit
import subprocess


classNames = {0: 'background',
			1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
			7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
			13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
			18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
			24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
			32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
			37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
			41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
			46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
			51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
			56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
			61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
			67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
			75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
			80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
			86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

# existing detector dunctions
class Detector:
			
	# vanilla model	
	def detectObject(self, imName):
		cvNet = cv.dnn.readNetFromTensorflow('model/object_detection/frozen_inference_graph.pb','model/object_detection/ssd_mobilenet_v1_coco_2017_11_17.pbtxt')
		img = cv.cvtColor(numpy.array(imName), cv.COLOR_BGR2RGB)
		cvNet.setInput(cv.dnn.blobFromImage(img, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
		detections = cvNet.forward()
		cols = img.shape[1]
		rows = img.shape[0]

		for i in range(detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.5:
				class_id = int(detections[0, 0, i, 1])

				xLeftBottom = int(detections[0, 0, i, 3] * cols)
				yLeftBottom = int(detections[0, 0, i, 4] * rows)
				xRightTop = int(detections[0, 0, i, 5] * cols)
				yRightTop = int(detections[0, 0, i, 6] * rows)

				cv.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
										 (0, 0, 255))
				if class_id in classNames:
					label = classNames[class_id] + ": " + str(confidence)
					labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
					yLeftBottom = max(yLeftBottom, labelSize[1])
					cv.putText(img, label, (xLeftBottom+5, yLeftBottom), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

		img = cv.imencode('.jpg', img)[1].tobytes()
		return img


	# retinanet model
	def run_detection_image(self, filepath):

		# read image
		image = read_image_bgr(filepath)

		# load model
		model_path = 'model/parkinglots_driveway/resnet50_csv_12_inference.h5'
		model = models.load_model(model_path, backbone_name='resnet50')
		labels_to_names = {0: 'parkinglot', 1: 'driveway'}


		# copy to draw on
		draw = image.copy()
		draw = cv.cvtColor(draw, cv.COLOR_BGR2RGB)

		# preprocess image for network
		image = preprocess_image(image)
		image, scale = resize_image(image)

		# process image
		start = time.time()
		boxes, scores, labels = model.predict_on_batch(numpy.expand_dims(image, axis=0))
		print("processing time: ", time.time() - start)

		# correct for image scale
		boxes /= scale

		# visualize detections
		for box, score, label in zip(boxes[0], scores[0], labels[0]):
			# scores are sorted so we can break
			if score < 0.5:
					break

			color = label_color(label)
			
			b = box.astype(int)
			draw_box(draw, b, color=color)

			caption = "{} {:.3f}".format(labels_to_names[label], score)
			draw_caption(draw, b, caption)


		file, ext = os.path.splitext(filepath)
		image_name = file.split('/')[-1] + ext
		output_path = os.path.join('examples/results/', image_name)
		
		draw_conv = cv.cvtColor(draw, cv.COLOR_BGR2RGB)
		img = draw_conv
		img = cv.imencode('.jpg', img)[1].tobytes()
		#cv.imwrite(output_path, draw_conv)

		return img

	"""
	def detect_COCO(self, file):

		print('here')
		cvNet_new = cv.dnn.readNetFromTensorflow('model/COCO/tflite_graph.pb', 'model/COCO/tflite_graph.pbtxt')
		print('here1')
		img = cv.imread(file)
		print('here2')
		rows = img.shape[0]
		cols = img.shape[1]
		print('here3')
		cvNet_new.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
		cvOut = cvNet_new.forward()
		print('here4')
		print(cols)
		print('here5')

		for detection in cvOut[0,0,:,:]:
			score = float(detection[2])
			if score > 0.3:
				left = detection[3] * cols
				top = detection[4] * rows
				right = detection[5] * cols
				bottom = detection[6] * rows
				cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)


		img = cv.imencode('.jpg', img)[1].tobytes()
		return img
	"""

class COCO:

	def load_labels(self, path, encoding='utf-8'):
		"""Loads labels from file (with or without index numbers).

		Args:
		path: path to label file.
		encoding: label file encoding.
		Returns:
		Dictionary mapping indices to labels.
		"""
		with open(path, 'r', encoding=encoding) as f:
			lines = f.readlines()
			if not lines:
				return {}

			if lines[0].split(' ', maxsplit=1)[0].isdigit():
				pairs = [line.split(' ', maxsplit=1) for line in lines]
				return {int(index): label.strip() for index, label in pairs}
			else:
				return {index: line.strip() for index, line in enumerate(lines)}


	def draw_objects(self, draw, objs, labels):
		"""Draws the bounding box and label for each object."""
		for obj in objs:
			bbox = obj.bbox
			draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],outline='red')
			draw.text((bbox.xmin + 10, bbox.ymin + 10),'%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),fill='red')


	def driver(self, file):
		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		parser.add_argument('-m', '--model', 
						  help='File path of .tflite file.', default = 'model/COCO/mobilenet_ssd_v1_coco_quant_postprocess.tflite')
		parser.add_argument('-i', '--input', 
						  help='File path of image to process.', default = file)
		parser.add_argument('-l', '--labels',
						  help='File path of labels file.', default = 'model/COCO/coco_labels.txt')
		parser.add_argument('-t', '--threshold', type=float, default=0.4,
						  help='Score threshold for detected objects.')
		parser.add_argument('-o', '--output',
						  help='File path for the result image with annotations', default = 'output.jpg')
		parser.add_argument('-c', '--count', type=int, default=5,
						  help='Number of times to run inference')
		args = parser.parse_args()

		print(args.labels)
		labels = self.load_labels(args.labels) if args.labels else {}
		interpreter = Interpreter(args.model)
		interpreter.allocate_tensors()

		image = Image.open(args.input).convert('RGB')
		scale = detect.set_input(interpreter, image.size,
							   lambda size: image.resize(size, Image.ANTIALIAS))

		print('----INFERENCE TIME----')
		for _ in range(args.count):
			start = time.monotonic()
			interpreter.invoke()
			inference_time = time.monotonic() - start
			objs = detect.get_output(interpreter, args.threshold, scale)
			print('%.2f ms' % (inference_time * 1000))

		print('-------RESULTS--------')
		if not objs:
			print('No objects detected')

		for obj in objs:
			print(labels.get(obj.id, obj.id))
			print('  id:    ', obj.id)
			print('  score: ', obj.score)
			print('  bbox:  ', obj.bbox)

		if args.output:
			self.draw_objects(ImageDraw.Draw(image), objs, labels)
		

		imgByteArr = io.BytesIO()
		image.save(imgByteArr, format='PNG')
		imgByteArr = imgByteArr.getvalue()

		return imgByteArr


class test_object:

	def cv_detect_object(file):

		print('here')
		cvNet_new = cv.dnn.readNetFromTensorflow('model/parkinglots_driveway/frozen_inference_graph.pb', 'model/parkinglots_driveway/graph.pbtxt')
		print('here1')
		img = cv.imread(file)
		print('here2')
		rows = img.shape[0]
		cols = img.shape[1]
		print('here3')
		cvNet_new.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
		cvOut = cvNet_new.forward()
		print('here4')
		print(cols)
		print('here5')

		for detection in cvOut[0,0,:,:]:
			score = float(detection[2])
			if score > 0.3:
				left = detection[3] * cols
				top = detection[4] * rows
				right = detection[5] * cols
				bottom = detection[6] * rows
				cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)


		img = cv.imencode('.jpg', img)[1].tobytes()
		return img

	def instagram_popularity(file):

		import argparse
		import torch
		import torchvision.models
		import torchvision.transforms as transforms
		from PIL import Image

		def prepare_image(image):
			if image.mode != 'RGB':
				image = image.convert("RGB")
			Transform = transforms.Compose([
					transforms.Resize([224,224]),      
					transforms.ToTensor(),
					])
			image = Transform(image)   
			image = image.unsqueeze(0)
			return image

		def predict(image, model):
			image = prepare_image(image)
			with torch.no_grad():
				preds = model(image)
			score = preds.detach().numpy().item()
			print('Popularity score: '+str(round(score,2)))

			return str(round(score,2))

		parser = argparse.ArgumentParser()
		parser.add_argument('--image_path', type=str, default=file)
		config = parser.parse_args()
		image = Image.open(config.image_path)
		model = torchvision.models.resnet50()
		# model.avgpool = nn.AdaptiveAvgPool2d(1) # for any size of the input
		model.fc = torch.nn.Linear(in_features=2048, out_features=1)
		model.load_state_dict(torch.load('model/instagram/model-resnet50.pth', map_location=torch.device('cpu'))) 
		model.eval()
		score = predict(image, model)

		return score



	# include self, file for python3
	def OCR_test(file, unique_name):

		from google.cloud import storage

		filename = unique_name+'.jpg'

		# upload
		bucket_name = 'push_bucket'
		source_file_name = file
		destination_blob_name = filename
		storage_client = storage.Client()
		bucket = storage_client.bucket(bucket_name)
		blob = bucket.blob(destination_blob_name)

		blob.upload_from_filename(source_file_name)

		print(
			"File {} uploaded to {}.".format(
				source_file_name, destination_blob_name
			)
		)

		# sleep
		time.sleep(3)

	
		# download 
		storage_client = storage.Client()
		file_data = filename+'_en.txt'
		bucket_name = 'spiyer99s_result_bucket'
		temp_file_name = filename+'_en.txt'
		bucket = storage_client.get_bucket(bucket_name)
		blob = bucket.get_blob(file_data)
		try:
			blob.download_to_filename(temp_file_name)
		except:
			# if file not found
			return ''
			#print(os.path.isfile('/app/'+filename+'_en.txt'))


	
		# delete files from result bucket
		bucket_name = "spiyer99s_result_bucket"
		storage_client = storage.Client()
		bucket = storage_client.bucket(bucket_name)
		languages = ['en', 'es', 'fr', 'ja', 'ru']
		for l in languages:
			blob_name = unique_name+'.jpg_'+l+'.txt'
			blob = bucket.blob(blob_name)
			try:
				blob.delete()
			except:
				pass

		# delete files from push bucket
		bucket_name = "push_bucket"
		storage_client = storage.Client()
		bucket = storage_client.bucket(bucket_name)
		blob_name = unique_name+'.jpg'
		blob = bucket.blob(blob_name)
		blob.delete()

		return '/app/'+filename+'_en.txt'


		# [START functions_ocr_detect]
	def detect_text(bucket, filename):
		print('Looking for text in image {}'.format(filename))

		futures = []

		text_detection_response = vision_client.text_detection({
			'source': {'image_uri': 'gs://{}/{}'.format(bucket, filename)}
		})
		annotations = text_detection_response.text_annotations
		if len(annotations) > 0:
			text = annotations[0].description
		else:
			text = ''
		print('Extracted text {} from image ({} chars).'.format(text, len(text)))

		detect_language_response = translate_client.detect_language(text)
		src_lang = detect_language_response['language']
		print('Detected language {} for text {}.'.format(src_lang, text))

		# Submit a message to the bus for each target language
		for target_lang in config.get('TO_LANG', []):
			topic_name = config['TRANSLATE_TOPIC']
			if src_lang == target_lang or src_lang == 'und':
				topic_name = config['RESULT_TOPIC']
			message = {
				'text': text,
				'filename': filename,
				'lang': target_lang,
				'src_lang': src_lang
			}
			message_data = json.dumps(message).encode('utf-8')
			topic_path = publisher.topic_path(project_id, topic_name)
			future = publisher.publish(topic_path, data=message_data)
			futures.append(future)
		for future in futures:
			future.result()
	# [END functions_ocr_detect]

