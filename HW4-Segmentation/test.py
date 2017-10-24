import sys
import os
import cv2


if __name__ == '__main__':

	root = os.path.dirname(os.path.realpath(__file__))
	print root

	if(sys.argv[1] == '1'):
		os.system("python "+root+"/Source/main.py "+root+"/TestData/astronaut.png "+root + "/TestData/astronaut_marking.png "+root+"/Results/")

		import imp
		RMSD = imp.load_source('main', root + '/Source/main.py')
		example_output = cv2.imread(root + "/example_output.png", cv2.IMREAD_GRAYSCALE)
		output = cv2.imread(root + "/Results/mask.png", cv2.IMREAD_GRAYSCALE)
		print "Error:", RMSD.RMSD(example_output, output)
	
	elif sys.argv[1] == '2':
		os.system("python "+root+"/Source/main_bonus.py "+root+"/TestData/astronaut.png")