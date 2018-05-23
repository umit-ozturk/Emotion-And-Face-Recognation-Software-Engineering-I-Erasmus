import argparse
import cv2
from time import sleep

training_path = "training"

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("Please Enter Positive Number")
    return ivalue


def capture_the_picture(number_of_photos, user):

	camera = cv2.VideoCapture(0)
	for x in range(number_of_photos+1):
	    return_value,image = camera.read()
	    cv2.imwrite(training_path + "/" + user + str(x) + '.jpg', image)
	    print( x + ". taked photo ")
	    sleep(2)
	camera.release()
	cv2.destroyAllWindows()



parser = argparse.ArgumentParser(description='Take picture for training data')
parser.add_argument("-p", "--photo", type=check_positive, default=10,
                    help="The number of photo for training")
parser.add_argument("-u", "--user", type=str, default="None",
                    help="The number of photo for training")
args = parser.parse_args()
print(args)


capture_the_picture(args.photo, args.user)
