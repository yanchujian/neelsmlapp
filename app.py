from ObjectDetector import Detector
from ObjectDetector import test_object
from ObjectDetector import COCO
import io
import subprocess
from flask import Flask, render_template, request, send_from_directory, send_file
from PIL import Image
from flask import send_file
import flask
import requests
import os
import urllib.request
import uuid 

#python2
#import urllib2.Request

app = Flask(__name__)
test = test_object()
detector = Detector()
coco = COCO()



#function to load img from url
def load_image_url(url):
	response = requests.get(url)
	img = Image.open(io.BytesIO(response.content))
	return img


@app.route("/")
def index():
	return render_template('index.html')


@app.route("/classify", methods=['POST', 'GET'])
def upload():
	if request.method == 'POST':
		file = Image.open(request.files['file'].stream)
		img = detector.detectObject(file)
		return send_file(io.BytesIO(img),attachment_filename='image.jpg',mimetype='image/jpg')
	elif request.method == 'GET':
		url = flask.request.args.get("url")
		print(url)
		file = load_image_url(url)
		img = detector.detectObject(file)
		return send_file(io.BytesIO(img),attachment_filename='image.jpg',mimetype='image/jpg')


@app.route("/parkinglot_detection", methods=['GET'])
def upload_parking():

	if request.method == 'GET':
		# get url
		url = flask.request.args.get("url")
		print(url)

		# save image in cwd
		filepath = os.getcwd()+'/file.jpg'
		urllib.request.urlretrieve(url, filepath)

		img = detector.run_detection_image(filepath)

		return send_file(io.BytesIO(img),attachment_filename='image1.jpg',mimetype='image1/jpg')


@app.route("/COCO_detection", methods=['GET'])
def upload_coco():

	if (request.method == 'GET'):

		# get url
		url = flask.request.args.get("url")
		print(url)

		# save image in cwd
		filepath = os.getcwd()+'/file.jpg'
		urllib.request.urlretrieve(url, filepath)

		# run detection
		img = coco.driver(filepath)

		return send_file(io.BytesIO(img),attachment_filename='image1.jpg',mimetype='image1/jpg')

@app.route("/insta", methods=['POST', 'GET'])
def insta():

	if request.method == 'POST':
		file = request.files['file']
		file.save(os.getcwd()+'/file.jpg')
		filepath = os.getcwd()+'/file.jpg'

	elif (request.method == 'GET'):

		# get url
		url = flask.request.args.get("url")
		print(url)

		# save image in cwd
		filepath = os.getcwd()+'/file.jpg'
		urllib.request.urlretrieve(url, filepath)


	# run detection
	score = test_object.instagram_popularity(filepath)

	text_file = open("output.txt", "w")
	text_file.write("Score: %s" % str(score))
	text_file.close()

	return send_file("output.txt", attachment_filename='filename.txt',mimetype='text/plain')


@app.route("/test", methods=['POST', 'GET'])
def test():

	#generate hash and get filepath
	unqiue_name = str(uuid.uuid1())
	filepath = os.getcwd()+'/'+unqiue_name+'.jpg'

	if request.method == 'POST':
		file = request.files['file']
		file.save(filepath)

	elif (request.method == 'GET'):

		# get url
		url = flask.request.args.get("url")
		print(url)

		# save image in cwd
		try:
			urllib.request.urlretrieve(url, filepath)
		except:
			# failure- invalid url
			return


	# run detection
	test_filepath = test_object.OCR_test(filepath, unqiue_name)
	# img = test_object.cv_detect_object(filepath)

	if(test_filepath == ''):
		return
	else:
		return send_file(test_filepath, attachment_filename='filename.txt',mimetype='text/plain')


if __name__ == "__main__":
	port = int(os.environ.get('PORT', 8080))
	#app.run()
	app.run(host='0.0.0.0', port=port)

