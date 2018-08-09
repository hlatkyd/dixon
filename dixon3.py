#!/usr/bin/python3

import sys
sys.path.append('/home/david/bin')
sys.path.append('/home/david/dev/common')
from readprocpar import procparReader
from readfdf import fdfReader

from writenifti import niftiWriter
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import glob

def load_data():
	# returns list of data

	def ifft(inkdata):

		inkdata = np.fft.fftshift(inkdata,axes=(2,3))
		ifft_data = np.fft.ifft2(inkdata,axes=(2,3), norm='ortho')
		ifft_data = np.fft.ifftshift(ifft_data,axes=(2,3))
		return ifft_data

	folder = '/home/david/dev/dixon/s_2018080901'
	name_list =  glob.glob(folder+'/fsems2*img')
	ind = [0,3,6]
	rawre = sorted([i for i in name_list if 'rawRE' in i ])
	rawim = sorted([i for i in name_list if 'rawIM' in i ])
	#print('\n'.join(rawim))
	#getting only the -pi, 0, pi
	combined_names = [[rawre[i],rawim[i]] for i in ind]
	data = []
	roshift = []
	for item in combined_names:
		ppr = procparReader(item[0]+'/procpar')
		roshift.append(float(ppr.read()['roshift']))
		hdr , data_re = fdfReader(item[0],'out').read()
		hdr , data_im = fdfReader(item[1],'out').read()
		cdata = np.vectorize(complex)(data_re[0,...],data_im[0,...])
		data.append(ifft(cdata))
	print(data[0].shape)
	print(roshift)

	return data, roshift

class dixon():

	def __init__(self,data,roshift,fieldmap=None):
		self.data = data
		self.roshift = roshift
	
	def ideal():
		"""
		F,S,RO : k th iter
		Fp,Sp,ROp : k+1 th iter 
		"""
		def fit(data1D, F0=0, freq=[0,1400]):
			"""
			1d function to be used for apply_along_axis
			F0 is initial field
			freq is the Dixon components frequency shift
			"""
			def init(data1D):
				Sre = np.asarray([np.real(data1D[i]) for i in range(data1D.shape[0])])
				Sim = np.asarray([np.imag(data1D[i]) for i in range(data1D.shape[0])])
				A = 0
				return A , np.concatenate(Sre, Sim)

			def ROp_from_S(S):
				pass
				return(ROp)
		
			def Sp_from_RO(RO):

				pass
				return Sp

			def Fp_from_S_Sp(S,Sp):
				pass
				return Fp, F

			point = init()

			return point

	
		#out = np.apply_along_axis(data,fit,axis=0)
		test = np.apply_along_axis(data,fit,axis=0)
		#return ro1_data, ro2_data
		return test

if __name__ == '__main__':
	data = load_data()
