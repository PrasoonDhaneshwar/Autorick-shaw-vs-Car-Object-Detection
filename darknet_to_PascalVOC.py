import os
import numpy as np
import glob
from PIL import Image
from time import time
from shutil import copy2

image_database_path = "/media/prasoon/DATA/tensorflow/models/research/object_detection/images_twoclass/Train/images"
tf_obj_imgdir_train = "Train"
tf_obj_imgdir_test = "Test"
tf_obj_imgpath = "/media/prasoon/DATA/tensorflow/models/research/object_detection/images_twoclass"
depth = 3
classes = ["autorickshaw","car"]

os.chdir(image_database_path)
np.random.seed(int(time()))
darknet_bb_files = glob.glob("*.txt")
rand_arr = np.random.uniform(0,1,len(darknet_bb_files))
thresh = 0.9

try: 
	os.mkdir(tf_obj_imgpath+tf_obj_imgdir_train)
except FileExistsError:
	print("Train folder already exists in %s" %(tf_obj_imgpath))
try: 
	os.mkdir(tf_obj_imgpath+tf_obj_imgdir_test)
except FileExistsError:
	print("Test folder already exists in %s" %(tf_obj_imgpath))

try: 
	os.mkdir(tf_obj_imgpath+tf_obj_imgdir_train+"/images")
except FileExistsError:
	print("Train folder already exists in %s" %(tf_obj_imgpath))
try: 
	os.mkdir(tf_obj_imgpath+tf_obj_imgdir_test+"/images")
except FileExistsError:
	print("Test folder already exists in %s" %(tf_obj_imgpath))
try: 
	os.mkdir(tf_obj_imgpath+tf_obj_imgdir_train+"/annotations")
except FileExistsError:
	print("Train folder already exists in %s" %(tf_obj_imgpath))
try: 
	os.mkdir(tf_obj_imgpath+tf_obj_imgdir_test+"/annotations")
except FileExistsError:
	print("Test folder already exists in %s" %(tf_obj_imgpath))

count = 0
for bb_file in darknet_bb_files:
	tf_obj_imgdir = tf_obj_imgdir_train
	if rand_arr[count] > thresh:
		tf_obj_imgdir = tf_obj_imgdir_test
	count += 1
	image_name = bb_file.split(".")[0] + ".JPEG"
	img = Image.open(image_name)
	copy2(image_name,tf_obj_imgpath + tf_obj_imgdir +"/images")
	w,h = img.size
	fp = image_database_path + bb_file
	fileR = open(fp,"r")
	lines = fileR.readlines()
	
	xmlname = bb_file.split(".")[0] + ".xml"
	xmlfp = tf_obj_imgpath + tf_obj_imgdir +"/annotations/"+ xmlname
	fileW = open(xmlfp,"w")
	fileW.write("<annotation>\n")
	fileW.write("\t<folder>"+tf_obj_imgdir+"</folder>\n")
	fileW.write("\t<filename>"+image_name+"</filename>\n")
	fileW.write("\t<path>"+tf_obj_imgpath+"</path>\n")
	fileW.write("\t<source>\n")
	fileW.write("\t\t<database>"+"Unknown"+"</database>\n")
	fileW.write("\t</source>\n")
	fileW.write("\t<size>\n")
	fileW.write("\t\t<width>"+str(w)+"</width>\n")
	fileW.write("\t\t<height>"+str(h)+"</height>\n")
	fileW.write("\t\t<depth>"+str(depth)+"</depth>\n")
	fileW.write("\t</size>\n")
	fileW.write("\t<segmented>"+"0"+"</segmented>\n")
	for line in lines:
		content = line.split(" ")
		yoloclass = classes[int(content[0])]
		cx = int(float(content[1])*w)
		cy = int(float(content[2])*h)
		pw = int(float(content[3])*w)
		ph = int(float(content[4])*h)
		xmin = int(cx - pw/2)
		ymin = int(cy - ph/2)
		xmax = int(cx + pw/2)
		ymax = int(cy + ph/2) 
		fileW.write("\t<object>\n")
		fileW.write("\t\t<name>"+yoloclass+"</name>\n")
		fileW.write("\t\t<pose>"+"Unspecified"+"</pose>\n")
		fileW.write("\t\t<truncated>"+"0"+"</truncated>\n")
		fileW.write("\t\t<difficult>"+"0"+"</difficult>\n")
		fileW.write("\t\t<bndbox>\n")
		fileW.write("\t\t\t<xmin>"+str(xmin)+"</xmin>\n")
		fileW.write("\t\t\t<ymin>"+str(ymin)+"</ymin>\n")
		fileW.write("\t\t\t<xmax>"+str(xmax)+"</xmax>\n")
		fileW.write("\t\t\t<ymax>"+str(ymax)+"</ymax>\n")
		fileW.write("\t\t</bndbox>\n")
		fileW.write("\t</object>\n")
	fileW.write("</annotation>\n")
	fileW.close()
	fileR.close()
	print("File %s written to %s " %(xmlname,tf_obj_imgdir))
