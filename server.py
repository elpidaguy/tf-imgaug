from flask import Flask, jsonify, redirect, request,Response, render_template, url_for, send_from_directory
import tensorflow as tf, sys
import base64
import os
import json
from werkzeug import secure_filename
from flask_cors import CORS, cross_origin
import hashlib
import math
from flask_uploads import UploadSet, configure_uploads
import requests

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
import numpy as np
import matplotlib.image as mpimg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


app = Flask(__name__)
CORS(app)
#app.config["MONGO_DBNAME"]="init6"
#mongo=PyMongo(app)
upload = "uploads/"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webupload',methods=['POST'])
def webupload():
	file = request.files['file']
	filename = secure_filename(file.filename)
	file.save(os.path.join(upload, filename))
	output_labels =[]
	final = os.path.join(upload, filename)
	Newlable = calculate(final,output_labels)
	#return Response("recived",mimetype='text/plain')
	#return jsonify({"Tree is":Newlable[0][0]})
	# results = mongo.db.plantInfo.find({"Botanical_Name":Newlable[0][0]},{"_id":False, "Location":False}).limit(1)
	# n_results = []
	# for res in results:
	# 	n_results.append(res)

	# locations = mongo.db.plantInfo.find({"Botanical_Name":Newlable[0][0]},{"_id":False, "Location":True})
	# locres = []
	# locList = []

	# # print(locations)
	# for recee in locations:
	# 	locList.append(recee)
	# for l in locList[0]["Location"]:
	# 	# print(l["State"])
	# 	locres.append([l["City"],l["Latitude"],l["Longitude"]])

		#for r in recee.values():
		#	print(r['State'])
		#locList.append([recee[0]["State"],recee["Latitude"],recee["Longitude"]])
	#return jsonify(locres[0])
	print (Newlable)
	return render_template("result.html", results = Newlable)

def calculate(final,output_labels):
	image_data = tf.gfile.FastGFile(final, 'rb').read()
	label_lines = [line.rstrip() for line in tf.gfile.GFile("Labels.txt")]
	with tf.gfile.FastGFile("Graph.pb", 'rb') as graphread:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(graphread.read())
		_ = tf.import_graph_def(graph_def, name='')
	with tf.Session() as sess:
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')			
		predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})			
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		# print(predictions)
		X = 0
		for node_id in top_k[:5]:
			X += predictions[0][node_id]
			# print(X)

		# print(X)
		for node_id in top_k[:5]:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			score1 = score / X
			#print(score*100)
			score1 = score1 * 100
			#print('%s (score = %.5f)' % (human_string, score))
			output_labels.append((human_string, str(round(score1,2))))
	# print(output_labels)
	return output_labels

@app.route('/augmentation')
def augmentation():
	return render_template('augmentation.html')

@app.route('/augupload',methods=['POST'])
def augupload():
	pass
	file = request.files['file']
	filename = secure_filename(file.filename)
	file.save(os.path.join(upload, filename))
	final = os.path.join(upload, filename)
	image = mpimg.imread(final)
	plt.imshow(image);
	plt.axis('off');
	plt.figure(figsize=(30,30))
	for i in range(10):
		img = transform_image(image,20,10,5,brightness=1)
		plt.imshow(img)
		plt.axis('off')
		plt.savefig('output/image'+str(i)+'.jpg')
	plt.show()
	return render_template('augmentation.html')

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness


    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img

if __name__ == '__main__':
	app.run(host='0.0.0.0',port=int(3535),debug=True, threaded=True)