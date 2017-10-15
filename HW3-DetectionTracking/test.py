import sys
import os

if __name__ == '__main__':
	qid = sys.argv[1]
	os.system("python Source/detection_tracking.py "+ str(qid) + " TestData/02-1.avi Results/")