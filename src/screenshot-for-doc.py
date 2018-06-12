import pyscreenshot as ImageGrab
from time import sleep
import os

savePath = "doc3"

# grab fullscreen
im = ImageGrab.grab()

i = 1
while True:
	try:
		os.stat(savePath)
	except:
		os.mkdir(savePath)
	# save image file
	im.save(savePath + "/" + str(i) + ".png")
	# show image in a window
	i = i + 1
	sleep(1)