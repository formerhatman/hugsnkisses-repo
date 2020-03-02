import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import wavio
import os
import simpleaudio as sa
import pandas as pd
from scipy.signal import stft,resample

def windowed_avg(a,size):
	fresh = np.arange(int(size/2),(a.shape[0] - int(size/2)) , step = size)
	bigolavgs = [np.mean(np.sqrt(a[i-int(size/2):i+int(size/2)]**2)) for i in fresh]
	# print(type(bigolavgs))
	return(np.asarray(bigolavgs))

def normalize_level(a):
	return(5*a/a.max())

def mean_freq(spectrum):
	meanfreq = 0
	summ = np.sum(spectrum)
	for f,amp in enumerate(spectrum):
		meanfreq+=(f*(amp))/summ
	return(meanfreq)

def max_freq(spectrum):
	return(np.where(spectrum==spectrum.max())[0])


def on_the_spectrum(data,new_calc=True):
	if new_calc:
		freqs = np.abs(np.fft.rfft(data))
	else:
		freqs=data
	if freqs.shape[0]< 21654:
		return(np.abs(np.pad(freqs,(0,21654-freqs.shape[0]),'constant',constant_values = 0.0)))
	else:
		return(np.abs(freqs[:21654]))


class Aud():
	def __init__(self,file,wind=400):
		self.directory = file
		self.sound = sa.WaveObject.from_wave_file(file)
		self.rate = wavio.read(file).rate
		self._raw = np.ravel(wavio.read(file).data)
		self.data = normalize_level(self._raw)
		# print(type(self.data))
		self.envelope = windowed_avg(self.data,wind)
		self.spectrum = np.abs(np.fft.rfft(self.data,n=21654))
	def play(self):
		self.sound.play()

	def better_spectrum(self,num_seg=75):
		res = int(np.round(self.data.shape[0]/(num_seg-1)))
		f,t,Zxx = stft(self.data,self.rate,nperseg=res,noverlap=0)
		new_env=resample(self.envelope,num_seg,window='blackmanharris')
		return(on_the_spectrum(resample(np.mean([new_env[i]*Zxx.T[i] for i in range(min(Zxx.shape[1],new_env.shape[0]))],axis=0),self.spectrum.shape[0],window='hann'),new_calc=False))
	def punch(self,num_seg=16):
		res = int(np.round(self.data.shape[0]/(num_seg)))
		fs = self.rate
		f,t,Zxx = stft(self.data,fs,nperseg=res,noverlap=res/2)
		freq_path=[]
		ft = np.abs(f)
		del_f = fs/res
		for seg in np.abs(Zxx.T):
			ind = np.where(seg==seg.max())
			if len(ind)>1:
				print('heyo')
			else:
				this_freq = ft[ind][0] / ft.max()
				freq_path.append(this_freq)
		return(normalize_level(np.asarray(freq_path[:33])))

old = 0


def Sound_Graph(x,y,sounds,names,dim = 2):
	if dim==3:
		pass
	else:
		fig = plt.figure()
		ax = fig.add_subplot(111)

		sc = ax.scatter(x,y)
		annots = []

		for i,name in enumerate(names):
		    annots.append(ax.text(x[i],y[i],names[i][:-4]))
		    # annots[i].get_bbox_patch().set_alpha(0.4)
		    annots[i].set_visible(False)
		def hover(event):
			global old
			if event.inaxes == ax:
				cont, ind = sc.contains(event)
				annots[old].set_visible(False)
				# print(cont,ind)
				if cont:
					ii=ind["ind"][0]
					sounds[ii].play()
					annots[ii].set_visible(True)
					fig.canvas.draw_idle()
					old = ii
				else:
					pass
		fig.canvas.mpl_connect("button_press_event", hover)
	return(fig,ax)