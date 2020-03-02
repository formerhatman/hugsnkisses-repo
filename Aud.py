import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import os
import simpleaudio as sa
import pandas as pd
from scipy.signal import stft,resample,resample_poly
import librosa as lb
from scipy.optimize import curve_fit
from math import floor,ceil
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider,Button
from soundfile import write
import pickle

def windowed_avg(a,size):
    fresh = np.arange(int(size/2),(a.shape[0] - int(size/2)) , step = size)
    bigolavgs = [np.mean(np.sqrt(a[i-int(size/2):i+int(size/2)]**2)) for i in fresh]
    # print(type(bigolavgs))
    return(np.asarray(bigolavgs))

def normalize_level(a):
    return(a/a.max())

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

def log_func(x,g,b,c):
            y_1=1/(1+np.exp((-1*x/g)-c))
            y_2=np.exp(-1*b*x)
            return(y_2 * (1-y_1))
class Aud():
    def __init__(self,file,dim=64):
        self.directory = file
        self.sound = sa.WaveObject.from_wave_file(file)
        self._raw,self.rate = lb.load(file)
        trimmed,ind = lb.effects.trim(self._raw,top_db=50)
        # print(index[0])
        if ((ind[1] + int(0.03*self.rate) - ind[0]) % 2):
            self.data = self._raw[ind[0]:ind[1] + int(0.03*self.rate) - 1]
        else:
            self.data = self._raw[ind[0]:ind[1] + int(0.03*self.rate)]
        wind = int(self.data.shape[0]/20)
        self.seg = floor(self.data.shape[0]/dim)
        self.length = self._raw.shape[0]
        self.envelope = resample(normalize_level(windowed_avg(self.data,wind)),64,window = 'hann')
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
    def env_peak(self,ind=False):
        peak_loc = np.where(self.envelope==self.envelope.max())[0][0]
        if ind:
            return(peak_loc)
        else:
            return(peak_loc/self.envelope.shape[0])
    def decay_log(self):
        peak_loc=self.env_peak(ind=True)
        # print(peak_loc)
        decay = self.envelope[peak_loc:]
        x = np.arange(self.envelope.shape[0]-peak_loc)
        # print(x)
        popt,pcov=curve_fit(log_func,x,decay,p0=[7,0.1,10],bounds=[[0,0,6],[50,50,10]])
        return(popt)


old = 0
def Sound_Graph_3D(data,sounds,names,slider = False):
    annots = []
    fig = plt.figure()
    if slider:
        x,y,z,slide_range=data
    else:
        try:
            x,y,z =data
        except ValueError:
            print('4-dimensional data requires slider=True!')
    ax=fig.add_subplot(111,projection='3d')
    sc = ax.scatter(x,y,z)
    for i,name in enumerate(names):
        annots.append(ax.text(x[i],y[i],z[i],names[i][:-4]))
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
    if slider:
        max_ax = plt.axes([0.25,0.1,0.65,0.03])
        min_ax = plt.axes([0.25,0.15,0.65,0.03])
        # butt_ax = plt.axes([0.8,0.025,0.1,0.04])
        max_slide = Slider(max_ax,'This one',slide_range.min(),slide_range.max(),valinit = slide_range.max(),valstep=slide_range.std())
        min_slide = Slider(min_ax,'That one',slide_range.min(),slide_range.max(),valinit = slide_range.min(),valstep=slide_range.std())
        # button = Button(butt_ax,'Update')
        def update(val):
            print('oy')
            sc.remove()
            low = min_slide.val
            high = max_slide.val
            print(low)
            mask = np.ma.masked_where(slide_range> low,slide_range)
            X = np.ma.masked_where(slide_range>low,x,np.nan)
            Y = np.ma.masked_where(slide_range>low,x,np.nan)
            Z = np.ma.masked_where(slide_range>low,x,np.nan)
            fig.canvas.draw_idle()
        max_slide.on_changed(update)
        min_slide.on_changed(update)
        # button.on_clicked(reset)
    return(fig,ax)

def Sound_Graph(x,y,sounds,names,nodes,dim = 2):
    annots = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(x,y)
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
    node_inds = [list(names).index(node) for node in nodes]
    for i in node_inds:
        annots[i].set_visible(True)
    return(fig,ax)

pca_df = pd.read_excel("C:/Users/Owner/Documents/Audio Processing Stuff/pca_data.xlsx",index_col=0)

def norm_df(df):
    for col in df:
        df[col] = ((df[col]-df[col].min())/(df[col].max()-df[col].min()))
    return(df)

def Spiderer(nodes,sounds,names,opendir="C:/Users/Owner/Documents/Audio Processing Stuff/WAAay Bigger Drumset/",savdir =  "C:/Users/Owner/Documents/Audio Processing Stuff/",new_data=False,save_data=False,num_pc=7):
    pcs = ['PC'+str(i) for i in range(1,num_pc+1)]
    meas = ['Amp','Freq','Punch'] + pcs
    cols = np.array([np.tile(np.array(nodes),len(meas)),np.repeat(meas,len(nodes))])
    MI = pd.MultiIndex.from_arrays(cols)
    data = np.zeros((len(names),len(nodes)*len(meas)))
    data_df = pd.DataFrame(data.T,index=MI).transpose()
    data_df.index =  names
    masters = [Aud(opendir + node) for node in nodes]
    if new_data:
        for i,file in enumerate(os.listdir(opendir)):
            if file in nodes:
                pass
            else:
                for node,master in zip(nodes,masters):
                    freq_diff = np.sum((sounds[i].spectrum - master.spectrum)**2)/21654
                    if master.envelope.shape[0]<sounds[i].envelope.shape[0]:
                        comp1 = np.pad(master.envelope,(0,sounds[i].envelope.shape[0]-master.envelope.shape[0]),'constant',constant_values=0.0)
                        comp2 = sounds[i].envelope
                    else:
                        comp1 = master.envelope
                        comp2 = np.pad(sounds[i].envelope,(0,master.envelope.shape[0]-sounds[i].envelope.shape[0]),'constant',constant_values=0.0)
                    amp_diff = np.sum((comp1 - comp2)**2)/comp1.shape[0]
                    punch_diff = np.sum((master.punch()-sounds[i].punch())**2)/16
                    for pc in pcs:
                        data_df.loc[file,(node,pc)]=np.abs(pca_df.loc[node,pc] - pca_df.loc[file,pc])
                    # punch_diff,let = pearsonr(sounds[i].punch(),master.punch())
                    data_df.loc[file,(node,'Amp')]=np.log(amp_diff)
                    data_df.loc[file,(node,'Freq')]=np.log(freq_diff)
                    data_df.loc[file,(node,'Punch')]=np.log(punch_diff)
    else:
        with open(savdir + 'data.pickle','rb') as f:
            data_dict = pickle.load(f)
        
        # names = os.listdir(opendir)
        
        masters_inds = [names.index(node) for node in nodes]
        
        masters_items = dict(zip(nodes,[dict(zip(meas,[data_dict[key][i] for key in meas])) for i in masters_inds]))
        
        for node,feat in data_df.columns.values:
            if feat in ['Amp','Freq','Punch']:
                func_over_form = lambda x: np.sum((masters_items[node][feat]-x)**2,axis=1)/x.shape[0]
            else:
                func_over_form = lambda x: x - masters_items[node][feat]
            new_data = func_over_form(np.asarray(data_dict[feat]))
            data_df[(node,feat)]=new_data
    data_df=norm_df(data_df)
    if save_data:
        
#         data_df.fillna(method='backfill',inplace=True)
        data_df.to_excel(savdir+'Spider_Data.xlsx')
    else:
        pass
    plotter = pd.DataFrame(np.ones((data_df.shape[0],2)),index = data_df.index,columns=['x','y'])
    theta = 2*np.pi / len(nodes)
    angles = np.array([i * theta for i in np.arange(len(nodes))])
    nodal_vectors = np.array([[np.cos(angle),np.sin(angle)] for angle in angles])
    reverse_vectors = np.array([[np.cos(angle+np.pi),np.sin(angle+np.pi)] for angle in angles])

    for wav in data_df.index.values:
        weights = np.array([[np.tile(data_df[node][feat][wav],2) for node in nodes] for feat in meas])
        # weights = [data_df[node][char][wav] for char in ['Freq','Amp','Punch']]
        pos = np.sum(np.multiply(np.sum(weights,axis=0),reverse_vectors)+nodal_vectors,axis=0)
        plotter.loc[wav]=pos
        # print(pos)
        # exit()

    return(plotter,nodal_vectors)

def sequencer(hits,tempo = 160):
    space_between = int((60/tempo)*22050/2)
    hits_len = len(hits)
    ii=0
    
    if hits_len > 8:
        lim = 8
    else:
        lim = 4
    while ((hits_len+ii)%lim)!=0:
        ii+=1
    # print(ii)
    empty_space=ii*space_between
    # last_hit = hits[-1]
    seq = np.zeros((len(hits)*space_between+empty_space),dtype=np.float32)
    seq_length = seq.shape[0]
    # print(seq_length)
    # seq = np.concatenate([hit for hit in hits])
    for i,hit in enumerate(hits):
        # print(hit.length)
        if i==hits_len-1:
            if hit.length>empty_space:
                seq = seq + np.pad(hit._raw[:empty_space],(i*space_between, seq_length - ((i*space_between)+empty_space)))
            else:
                seq = seq + np.pad(hit._raw,(i*space_between, seq_length - ((i*space_between)+hit.length)))
        else:
            seq = seq + np.pad(hit._raw,(i*space_between, seq_length - ((i*space_between)+hit.length)))
    seq = np.tile(seq,2)
    write("C:/Users/Owner/Documents/Audio Processing Stuff/TEMP/tempwav.wav",seq.astype(np.float32),22050)
    seq_sound = sa.WaveObject.from_wave_file("C:/Users/Owner/Documents/Audio Processing Stuff/TEMP/tempwav.wav")
    seq_sound.play()
    return(seq)


# nodes =['Reverb Korg Rhythm 55 Sample Pack_Rim.wav','CR78_Rim_Shot.wav','Reverb E-mu Drumulator Sample Pack_Clap.wav','CR78_HiHat_Metal.wav','LongTriangle.wav','ASR-X Hat 14.wav']
# opendir="C:/Users/Owner/Documents/Audio Processing Stuff/WAAay Bigger Drumset/"
# hits = [Aud(opendir+wav) for wav in nodes]
# seq = sequencer(hits)

# plt.plot(seq[::10000])
# seq_sound = Aud("C:/Users/Owner/Documents/Audio Processing Stuff/TEMP/tempwav.wav")
# seq_sound.play()