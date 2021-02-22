# USAGE
# python cluster_faces.py --encodings encodings.pickle

# import the necessary packages
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
from datetime import datetime
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of parallel jobs to run (-1 will use all CPUs)")
ap.add_argument("-o", "--output", default=str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")), #type=int, default=-1,
	help="# of parallel jobs to run (-1 will use all CPUs)")
	
args = vars(ap.parse_args())


if not os.path.exists(args["output"]):
			os.makedirs(args["output"])
		

# load the serialized face encodings + bounding box locations from
# disk, then extract the set of encodings to so we can cluster on
# them
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]
#global face
#face = []
# cluster the embeddings
print("[INFO] clustering...")
clt = DBSCAN(metric="euclidean", n_jobs=args["jobs"])
clt.fit(encodings)

# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

global contatore
contatore = 0

global contatores
contatores = 0

oggi = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
				
if not os.path.exists(oggi):
	os.makedirs(oggi)


# loop over the unique face integers
for labelID in labelIDs:
	contatore =(contatore+1)
		
	# find all indexes into the `data` array that belong to the
	# current label ID, then randomly sample a maximum of 25 indexes
	# from the set
	print("[INFO] faces for face ID: {}".format(labelID))
	
	idxs = np.where(clt.labels_ == labelID)[0]
	idxs = np.random.choice(idxs, size=min(25, len(idxs)),
		replace=False)

	# initialize the list of faces to include in the montage
	faces = []
	
	if not os.path.exists(oggi+'/'+str(contatore)):
			os.makedirs(oggi+'/'+str(contatore))


	# loop over the sampled indexes
	for i in idxs:
		
		contatores = (contatores+1)
		# load the input image and extract the face ROI
		image = cv2.imread(data[i]["imagePath"])
		print(str(data[i]["imagePath"]))
		immagine = cv2.imread(str(data[i]["imagePath"]))
		#immagine = cv2.resize(image, (96, 96))
		cv2.imshow(str(contatores) , immagine)  ### aggiunto
		new = str(data[i]["imagePath"]).replace("/", "_")
		cv2.imwrite(oggi+'/'+str(contatore)+'/'+new, immagine)
		try:
			(top, right, bottom, left) = data[i]["loc"]
			face = image[top:bottom, left:right]

			# force resize the face ROI to 96x96 and then add it to the
			# faces montage list
			face = cv2.resize(face, (96, 96))
			faces.append(face)
		except:
				
			continue

	# create a montage using 96x96 "tiles" with 5 rows and 5 columns
	montage = build_montages(faces, (96, 96), (5, 5))[0]
	
	# show the output montage
	title = "Face ID #{}".format(labelID)
	title = "Unknown Faces" if labelID == -1 else title
	cv2.imshow(title, montage)
	cv2.imwrite(args["output"]+'/'+str(title)+'.png', montage)
	cv2.waitKey(0)
