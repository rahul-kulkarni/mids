# import all packages
# n.b. VLC needs to be open BEFORE running the program
import pyopenpose as op
from imutils import translate, rotate, resize
import time
import numpy as np
np.random.seed(1337)

import tensorflow as tf
import keras
import pafy
import vlc
import cv2

# make sure tensorflow doesn't take up all the gpu memory
conf = tf.ConfigProto()
conf.gpu_options.allow_growth=True
session = tf.Session(config=conf)

# Custom Params (refer to include/openpose/flags.hpp for more parameters).  May need to modify on different machines.
params = dict()
params["model_folder"] = "./openpose/models/"

# Use USB camera (TX2  isn't working well)
vs = cv2.VideoCapture(1)
#time.sleep(2)

# This script is currently run in the parent directory of dab-and... May need to modify on different machines.
tposer = keras.models.load_model('./dab-and-tpose-controlled-lights/data/dab-tpose-other.h5')

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
np.set_printoptions(precision=4)

fps_time = 0

# Could be expanded upon in future iterations
DAB = 1
TPOSE = 2
OTHER = 0

bounced = time.time()
debounce = 3# wait 3 seconds before allowing another command

while True:
    ret_val, frame = vs.read()

    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # need to be able to see what's going on
    image = datum.cvOutputData

    cv2.putText(image,
                "FPS: %f" % (1.0 / (time.time() - fps_time)),
                (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)

    cv2.imshow("Openpose", image)
    cv2.waitKey(50)

    if datum.poseKeypoints.any():
        first_input = datum.poseKeypoints
        try:
            first_input[:,:,0] = first_input[:,:,0] / 720
            first_input[:,:,1] = first_input[:,:,1] / 1280
            first_input = first_input[:,:,1:]
            first_input = first_input.reshape(len(datum.poseKeypoints), 50)
        except:
            continue

        # urls are static for this example, but need not be going forward
        output = tposer.predict_classes(first_input)
        for j in output:
            if j == 1:
                print("dab detected")
                url = "https://www.youtube.com/watch?v=fveEzcAkPnM"# "Look At My Dab" (Chipmunk Version, standard was not appropriate for a classroom)
                video = pafy.new(url)
                best = video.getbestaudio()
                player = vlc.MediaPlayer(best.url)
                player.stop()# per documentation, has no effect if not being played
                player.play()
                #time.sleep(10)
                break

            elif j == 2:
                print("tpose detected")
                url = "https://www.youtube.com/watch?v=nfWlot6h_JM"# "Shake It Off" by Taylor Swift (T-Pose == T-Swift)
                video = pafy.new(url)
                best = video.getbestaudio()
                player = vlc.MediaPlayer(best.url)
                player.stop()# per documentation, has no effect if not being played
                player.play()
                #time.sleep(10)
                break

    fps_time = time.time()

    # quit with a q keypress, b or m to save data
    key = cv2.waitKey(1)
    if key == ('q'):
        break

# clean up after yourself
vs.release()
cv2.destroyAllWindows()