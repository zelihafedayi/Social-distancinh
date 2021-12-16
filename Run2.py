from mylib import config, thread
from mylib.mailer import Mailer
from mylib.detection import detect_people
from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist
import numpy as np
import argparse, imutils, cv2, os, time, schedule
import sqlite3
from mylib.database import People
import string
import datetime
#from mylib.graph import Graph

conn = sqlite3.connect('people_database.db')
conn.row_factory = sqlite3.Row # This enables column access by name: row['column_name']
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS people (
            n text,
			s text,
            h text,
            m text,
            t1 text,
			t2 text,
			c text
            )""")

def timestamp():
	now=datetime.datetime.utcnow()
	return now.strftime('%Y-%m-%dT%H:%M:%S') + now.strftime('.%f')[:4] #+ 'UTC 0'#datetime.datetime.utcnow().isoformat() + 'Z' #timespec='milliseconds'

def detect():
	writer = None
	# start the FPS counter
	fps = FPS().start()
	while True:
		# read the next frame from the file
		if config.Thread:
			frame = cap.read()

		else:
			(grabbed, frame) = vs.read()
			# if the frame was not grabbed, then we have reached the end of the stream
			if not grabbed:
				break

		# resize the frame and then detect people (and only people) in it
		frame = imutils.resize(frame, width=700)
		results,total_people = detect_people(frame, net, ln,
			personIdx=LABELS.index("person"))

		# initialize the set of indexes that violate the max/min social distance limits
		serious = set()
		abnormal = set()
		#print(total_people)
		# ensure there are *at least* two people detections (required in
		# order to compute our pairwise distance maps)
		if len(results) >= 2:
			# extract all centroids from the results and compute the
			# Euclidean distances between all pairs of the centroids
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

			# loop over the upper triangular of the distance matrix
			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					# check to see if the distance between any two
					# centroid pairs is less than the configured number of pixels
					if D[i, j] < config.MIN_DISTANCE:
						# update our violation set with the indexes of the centroid pairs
						serious.add(i)
						serious.add(j)
	                # update our abnormal set if the centroid distance is below max distance limit
					if (D[i, j] < config.MAX_DISTANCE) and not serious:
						abnormal.add(i)
						abnormal.add(j)

		# loop over the results
		for (i, (prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)

			# if the index pair exists within the violation/abnormal sets, then update the color
			if i in serious:
				color = (0, 0, 255)
			elif i in abnormal:
				color = (0, 255, 255) #orange = (0, 165, 255)

			# draw (1) a bounding box around the person and (2) the
			# centroid coordinates of the person,
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 2)

		# draw some of the parameters
		Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
		cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
		Threshold = "Threshold limit: {}".format(config.Threshold)
		cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
			cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

	    # draw the total number of social distancing violations on the output frame
		text = "Total serious violations: {}".format(len(serious))
		cv2.putText(frame, text, (10, frame.shape[0] - 55),
			cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

		text1 = "Total abnormal violations: {}".format(len(abnormal))
		cv2.putText(frame, text1, (10, frame.shape[0] - 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

	#------------------------------Alert function----------------------------------#
		if len(serious) >= config.Threshold:
			cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
				cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
			if config.ALERT:
				print("")
				print('[INFO] Sending mail...')
				Mailer().send(config.MAIL)
				print('[INFO] Mail sent')
			#config.ALERT = False
	#------------------------------------------------------------------------------#
		# check to see if the output frame should be displayed to our screen
		if args["display"] > 0:
			# show the output frame
			cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
			if key == ord("n"):
				new_person()
			if key == ord('e'):
				person_exit()
				#break
				#new=input("enter name and hes code: ")
				#print("welcome")
				#print("new")
				#print("checking hes code")
	    # update the FPS counter
		fps.update()

		# if an output video file path has been supplied and the video
		# writer has not been initialized, do so now
		if args["output"] != "" and writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 25,
				(frame.shape[1], frame.shape[0]), True)

		# if the video writer is not None, write the frame to the output video file
		if writer is not None:
			writer.write(frame)
		fps.stop()
		#print("===========================")
		#print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
		#print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

def new_person():
    now=datetime.datetime.utcnow()
    new=input("if it is your first time please type your name, surname, hes code and e-mail address, if not please type your name and surname: ")
	#print(new)
    entry = new.split(' ')
    print(len(entry))
    if (len(entry)==3): #special conditions
        if new.upper().startswith("CP"): # Covid patient E.g. CP name surname
            cpatient= entry[1]
            print(cpatient)
			#c='1'
            print("Etrance is  forbidden!")
			#take name surname time (check_patient)
			#timestamp=now.strftime('%Y-%m-%dT%H:%M:%S') + now.strftime('.%f')[:4] + 'Z'
			#timestamp2=now.strftime('%Y-%m-%dT%H:%M:%S') + now.strftime('.%f')[:4] + 'Z'
			#newdata=People('koray','aydın','2323','dfsd',timestamp,timestamp2,c)
			#insert_to_db(newdata)
            new_patient(cpatient)
            check_patient()
			#------------mailalert-----------
        elif new.upper().startswith("HP"): #Healty person E.g. HP name surname
            hpatient= entry[1]
            healty(hpatient)
            check_patient()
    elif (len(entry)==2):#get mail and hes info from db and insert new input to the database
        pmail,phes=get_info_by_name(entry[0],entry[1])
        if (pmail=="notfound"):
            print("user not found")
            user_input_info()
        else:
            c='0'
            name=entry[0]
            print("Welcome " + name)
            surname=entry[1]
            print(pmail)
            print(phes)
            timestamp1=timestamp() #now.strftime('%Y-%m-%dT%H:%M:%S') + now.strftime('.%f')[:4] + 'UTC 0'
            timestamp2=timestamp1
            newdata=People(name,surname,phes,pmail,timestamp1,timestamp2,c)
            insert_to_db(newdata)#print("thanks: " + name)
            user_input_info()

    elif (len(entry)<2):
        print("Missing information!")
        user_input_info()

    elif (len(entry) > 3):
        c='0'
        name= entry[0]
        print("thanks " + name)
        user_input_info()
        surname= entry[1]
		#print("surname: " + surname)
        hes= entry[2]
        #time_entrance=timestamp()
        mail = entry[3]
        timestamp1=timestamp() #now.strftime('%Y-%m-%dT%H:%M:%S') + now.strftime('.%f')[:4] + 'UTC 0'
        timestamp2=timestamp1
        newdata=People(name,surname,hes,mail,timestamp1,timestamp2,c)
        insert_to_db(newdata)#print("thanks: " + name)


def user_input_info():
    #print every new user input
    print("---To enter place type N on the frame window--- \n---To exit the place type E on the frame window--- \n---To close the program please press Q on the frame window")

def insert_to_db(people):
	with conn:
		c.execute("INSERT INTO people VALUES (:n, :s, :h, :m, :t1, :t2, :c)", {'n': people.n, 's': people.s, 'h': people.h, 'm': people.m, 't1':people.t1, 't2':people.t2, 'c':people.c})
def delete_old_data():
    tnow=timestamp()
    timewithmoths=tnow.split('T')[0]
    d=timewithmoths.split('-')[2]
    dint=int(d)
    with conn:
        rows=c.execute("""SELECT rowid,t1 FROM people""")
        data=rows.fetchall()
        for datas in data:
            row_id= datas[0]
            row_time=datas[1]
            day=(row_time.split('T')[0]).split('-')[2]
            print(day)
            dayint=int(day)
            if dayint<dint and dint-dayint>=config.MAX_DAY:
                c.execute("DELETE from people WHERE rowid = :rowid",{'rowid': row_id})
            if dayint>dint and 30-dayint+dint>=config.MAX_DAY:
                c.execute("DELETE from people WHERE rowid = :rowid",{'rowid': row_id})
            #print(row_id)
            #print(row_time)
def check_patient():
	with conn:
		rows=c.execute("""SELECT n,s,m FROM people WHERE c='1' ORDER BY t2 ASC""")
		data=rows.fetchall()
		for datas in data:
			cp_mail= datas[2]
			print(cp_mail)
		print(cp_mail)
        #alert(mail)

def new_patient(n): #change person to unhealty condition
	with conn:
		c.execute("""UPDATE people SET c='1' WHERE n= :n AND s=:s """,	{'n':n, 's':s}) # update healt by name and surname
		#data=rows.fetchall()
		#for datas in data:
			#print(datas[1])
def healty(n): #change to healty condition
	with conn:
		c.execute("""UPDATE people SET c='0' WHERE n= :n AND s=:s """,	{'n':n, 's':s})

#def send_alert_mail():
def person_exit():
    now=datetime.datetime.utcnow()
    t2=now.strftime('%Y-%m-%dT%H:%M:%S') + now.strftime('.%f')[:4] #+ 'UTC0'
    new=input("Please type your name and surname: ")
    exit_p=new.split(' ')
    #print(len(exit_p))
    if (len(exit_p)<2):
        print("Incorrect information. E.g. dilay özden")
    else:
        n=exit_p[0]
        print(n)
        s=exit_p[1]
        print(s)
        print("Exit time: " + t2)
        with conn:
    	       c.execute("""UPDATE people SET t2=:t2 WHERE n=:n AND s=:s AND t1=t2""",	{'t2':t2, 'n':n, 's':s})
        user_input_info()


def get_info_by_name(name,surname):
    try:
        rows=c.execute("SELECT h,m FROM people WHERE n=:n", {'n': name})
        data=rows.fetchall()
        for datas in data:
            p_mail= datas[1]
            p_hes=datas[0]
            #print(p_mail)
        #print(p_hes)
        return p_mail, p_hes
    except:
        name="notfound"
        surname="notfound"
        return name, surname
#----------------------------Parse req. arguments------------------------------#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())
#------------------------------------------------------------------------------#

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("")
	print("[INFO] Looking for GPU")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# if a video path was not supplied, grab a reference to the camera
if not args.get("input", False):
	print("[INFO] Starting the live stream..")
	vs = cv2.VideoCapture(config.url)
    #user_input_info()
	if config.Thread:
			cap = thread.ThreadingClass(config.url)
	time.sleep(2.0)
    #user_input_info()

# otherwise, grab a reference to the video file
else:
	#print("[INFO] Starting the video..")
	vs = cv2.VideoCapture(args["input"])
	if config.Thread:
			cap = thread.ThreadingClass(args["input"])



# loop over the frames from the video stream
#while True:
#input = input("Enter command: ")
#print(input)
#timenow=timestamp()
#Mailer().send_max_people(config.MAIL,timenow)
delete_old_data()
user_input_info()
detect()

# stop the timer and display FPS information


# close any open windows
cv2.destroyAllWindows()
