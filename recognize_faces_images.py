# USAGE
# python3.6 recognize_faces_image.py --encodings encodings.pickle2 --images target/ 
# python3.6 recognize_faces_image.py --encodings encodings.pickle2 --images target/


# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import numpy as np
from datetime import datetime
import os
import hashlib
				
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default="encodings.pickle2",#required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--images", default="target",#required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
datas = pickle.loads(open(args["encodings"], "rb").read())
data = np.array(datas)

dats = [d["encoding"] for d in data]
nomes = [d["imagePath"] for d in data]
print(nomes[2])
# load the input image and convert it from BGR to RGB

oggi = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
				
if not os.path.exists(oggi):
	os.makedirs(oggi)


images = os.listdir(args["images"])
global unico
uno = 0

for immagine in images:
	uno = uno+1
	unico = str(uno)
	if not os.path.exists(oggi+'/'+unico):
		os.makedirs(oggi+'/'+unico)

	
	print(str(immagine))
	image = cv2.imread(args["images"]+'/'+str(immagine))
	

	
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	print("[INFO] recognizing faces...")
	boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
	encoding = face_recognition.face_encodings(rgb, boxes)

	# cluster the embeddings
	#print("[INFO] clustering...")
	#clt = DBSCAN(metric="euclidean", n_jobs=args["jobs"])
	#clt.fit(dats)

	# determine the total number of unique faces found in the dataset
	#labelIDs = np.unique(clt.labels_)
	#numUniqueFaces = len(np.where(labelIDs > -1)[0])
	#print("[INFO] # unique faces: {}".format(numUniqueFaces))



	# initialize the list of names for each face detected
	names = []
	global contatore
	contatore = int(0)
	global name
	name = "Unknown"

	global contatores
	contatores = 0
	###### CALCOLO HASH #######
	
	
	file1 = (str(args["images"])+'/'+str(immagine))      #,"r", encoding='utf-8')
	openFile1 = open(file1, "rb")
	readFile1 = openFile1.read()
	
	md5hash1 = hashlib.md5(readFile1)
	md5file1 = md5hash1.hexdigest()
	
	sha1hash1 = hashlib.sha1(readFile1)
	shafile1 = sha1hash1.hexdigest()
	
	openFile1.close()
	
	############################

	csv = open(oggi+'/'+unico+'/risultato.csv','a')
	# loop over the facial embeddings
	
	
	csv.write("0 immagine ricercata ;"+str(args["images"])+'/'+str(immagine)+';'+str(md5file1)+';'+str(shafile1)+'\n')# args["images"]+'/'+str(immagine)
	for dat in dats:
		contatore = (contatore+1)
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(encoding, dat)
	

		# check to see if we have found a match
		if True in matches:
			print("===")
			print(str(nomes[contatore-1]))
			print("===")
				# find the indexes of all matched faces then initialize a
				# dictionary to count the total number of times each face
				# was matched
			name = str(nomes[contatore-1])
#				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
#				counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
#			for i in matchedIdxs:
#				name = dat["ID"][i]
#				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number of
			# votes (note: in the event of an unlikely tie Python will
			# select first entry in the dictionary)
			#name = max(counts, key=counts.get)
	
		# update the list of names
			names.append(name)

# 	loop over the recognized faces
			for ((top, right, bottom, left), name) in zip(boxes, names):
				# draw the predicted face name on the image
				cv2.rectangle(image, (left, top), (right, bottom), (220, 0, 0), 1)
				y = top - 15 if top - 15 > 15 else top + 15
				cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
					0.75, (225, 0, 0), 2)
				
			target = cv2.imread(str(nomes[contatore-1]))
			cv2.imshow(str(nomes[contatore-1]), target)
			cv2.imwrite(oggi+'/'+unico+'/'+str(contatore-1)+'.png', target)
			contatores = contatores+1
			########## calcolo HASH #######
			file1 = (str(nomes[contatore-1]))      #,"r", encoding='utf-8')
			openFile1 = open(file1, "rb")
			readFile1 = openFile1.read()
	
			md5hash1 = hashlib.md5(readFile1)
			md5file1 = md5hash1.hexdigest()
	
			sha1hash1 = hashlib.sha1(readFile1)
			shafile1 = sha1hash1.hexdigest()
	
			openFile1.close()
			###############################
			csv.write(str(contatores)+' immagine matching '+';'+str(nomes[contatore-1])+';'+str(md5file1)+';'+str(shafile1)+'\n')
			cv2.waitKey(0)
	csv.close()
	# show the output image
	cv2.imshow("immagine ricercata"+"file:  "+args["images"]+'/'+str(immagine), image)
	cv2.imwrite(oggi+'/'+unico+'/immagine_ricercata_'+args["images"]+'/'+str(immagine), image)
	cv2.waitKey(0)
exit
