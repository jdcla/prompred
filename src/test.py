import numpy as np
import sys
import argparse


def printJim(times,string,lists):
	print(np.str(string) == 'jim')
	print(lists)
	if times is not None:
		print(string*times)

def main():
	parser = argparse.ArgumentParser(description='high-end script function for prompred')
	parser.add_argument('-n', type=int, default=None, help='number of times to print Jim')
	parser.add_argument('string', choices=('jim','tom'), help='add str')
	parser.add_argument('list1', nargs='*', type=int, help='add str')
	args = parser.parse_args()
	printJim(args.n, args.string, args.list1)



if __name__ == "__main__":
    sys.exit(main())
