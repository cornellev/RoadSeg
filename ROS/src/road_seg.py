# import the necessary packages
import numpy as np
import imutils
import cv2
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge as bridge
import time


FILEPATH = "/home/cev/catkin_ws/src/road_seg/src/enet-model.net" #fill with Local filepath to model parameters
CLASSES = open("/home/cev/catkin_ws/src/road_seg/src/class.txt","r") #fill with Local filepath to class params
colors = open("/home/cev/catkin_ws/src/road_seg/src/color.txt","r").read().strip().split("\n") #fill with Local filepath to color params (used for object vs. road vs. undetected)
name = "/home/cev/catkin_ws/src/road_seg/src/vis.txt"
VIS = open(name, "r")
vis = VIS.read().strip().split("\n")
vis = [np.array(c.split(",")).astype("int") for c in colors]
vis = np.array(colors, dtype="uint8")

classes = CLASSES.read().strip().split("\n")
net = cv2.dnn.readNet(FILEPATH)
print("predicted")
legend = np.zeros(((len(classes) * 25) + 25, 300, 3), dtype="uint8")
for (i, (className, color)) in enumerate(zip(classes, colors)):
	# draw the class name + color on the legend
	color = [int(c) for c in color]
	cv2.putText(legend, className, (5, (i * 25) + 17),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
		tuple(color), -1)
print("color setup")

def seg_callback(data):
    print("called")
    image = bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    image = imutils.resize(image, width=500)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0,
	swapRB=True, crop=False)
    net.setInput(blob)
    print("set input")
    start = time.time()
    output = net.forward()
    end = time.time()
    print("[INFO] inference took {:.4f} seconds".format(end - start))
    
    classMap = np.argmax(output[0], axis=0)
    mask = colors[classMap]
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
	interpolation=cv2.INTER_NEAREST)
    classMap = cv2.resize(classMap, (image.shape[1], image.shape[0]),
	interpolation=cv2.INTER_NEAREST)
    output = ((0.4 * image) + (0.6 * mask)).astype("uint8")
    #print would go below here
    cv2.imshow(output)
    pub.publish(True)

if __name__ == '__main__':
    while 1:
        try:
            print(1)
            rospy.init_node('RSeg', anonymous=True)
            print(2)
            pub = rospy.Publisher('RSeg', Bool, queue_size=1)
            print(3)
            depthsub = rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, seg_callback)
            print(4)
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
