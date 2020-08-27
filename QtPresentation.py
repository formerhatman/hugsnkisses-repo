# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 13:28:07 2020

@author: Owner
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import Aud as hk
from math import ceil,floor
from sklearn.decomposition import FastICA,PCA
import os
from sklearn.manifold import TSNE
import pandas as pd
from librosa import get_samplerate
import pickle
from PyQt5 import QtGui,QtCore
import pyqtgraph as pg
from wave import Error
'''
important backend files:
    HnKfile_list.dat
    HnKdata.pickle
    HnK_melspec.pickle
    HnK_paths.dat
    
'''
def get_entries(filename):
    with open(filename,'r') as f:
        entries = f.readlines()
    entries = np.array([entry[:-1] for entry in entries])
    return(entries)


def gridsupport(num_points):
    start = np.sqrt(num_points)
    x = ceil(start)
    y = floor(start)
    return([x,y])
    
def iswav(a):
    if type(a)=='numpy.ndarray':
        pass
    else:
        a = np.array(a)
    return(a[['.wav' in file.lower()[-4:] for file in a]])


def make_file_list():
    dirdir = "C:/Users/Owner/Documents/Audio Processing Stuff/WAAay Bigger Drumset/"
    with open(os.getcwd()+'\\file_list.dat','w') as f:
        for file in os.listdir(dirdir):
            f.write(dirdir + file + '\n')

def get_waves(groot):
    file_list = []
    for root, dirs, files in os.walk(groot):
        newfiles = [root + '/' + file for file in files]
    file_list = file_list + newfiles
    return(iswav(file_list))

opendir1 =  "C:/Users/Owner/Documents/Audio Processing Stuff/WAAay Bigger Drumset/"
opendir2 = "C:/Users/Owner/Documents/Audio Processing Stuff/[KB6]_Korg_DDD-1/"
opendir3 = "C:/Users/Owner/Documents/Audio Processing Stuff/[KB6]_Casio_Rapman/"

savdir =  "C:/Users/Owner/Documents/Audio Processing Stuff/"

data_files = ['\\HnK_paths.dat','\\HnKfile_list.dat','\\HnKdata.pickle','\\HnK_melspec.pickle','\\HnKsounds.pickle']


class data_format():
    def __init__(self,paths,label):
        self.opendirs = paths
        self.label = label
        self.check_files()
    def check_files(self):
        self.filecheck = all([os.path.exists(os.getcwd()+file) for file in data_files])
        print(self.filecheck)

    
    def set_label(self,label):
    	self.label = label
    
    def print_to_label(self,string):
        self.label.setText(string)
        print(string)
    
    def check_newbies(self):
        if self.filecheck==False:
            self.refresh_firsttime()
            return
        else:
            with open(os.getcwd() + '\\HnKfile_list.dat','r') as f:
                old_file_list = np.array([file[:-1] for file in f.readlines()])
                f.close()
        new_files = np.array([])
        for opendir in self.opendirs:
            new_files = np.concatenate([new_files, get_waves(opendir)])
        if np.setdiff1d(new_files, old_file_list).shape[0] != 0:
            print('new files!')
            new_files = np.setdiff1d(new_files, old_file_list)
            self.refresh_newsounds(new_files)
            return
        else:
            print('no new sounds')
            self.refresh_nonew()
            return
    def refresh_nonew(self):
        print('no new sounds.')
        with open(os.getcwd() + '\\HnKdata.pickle', 'rb') as f:
            data = pickle.load(f)
        self.data = data
        with open(os.getcwd() + '\\HnK_melspec.pickle', 'rb') as f:
            melspec = pickle.load(f)
        self.PCAaay(melspec)
        with open(os.getcwd() + '\\HnKfile_list.dat', 'r') as f:
            old_file_list = [file[:-1] for file in f.readlines()]
            f.close()
        self.files = old_file_list
        with open(os.getcwd() + '\\HnKsounds.pickle', 'rb') as f:
            self.sounds = pickle.load(f)
            f.close()
        self.indicies = np.arange(len(self.files))
    def refresh_newsounds(self,new_names):
        print('new sounds!')
        new_sounds = []
        true_names = []
        for name in new_names:
            try:
                new_sounds.append(hk.Soundo(name))
                true_names.append(name)
            except Error:
                print(get_samplerate(name))
                pass
        '''data format:
            [Envelope,FFT,Punch,melspec]
            PCA is pretty fast, so it's no big deal to run it again.
        '''
        if len(true_names)==0:
            self.refresh_nonew()
            return
        new_envs = []
        new_spectrums = []
        new_punches = []
        new_melspecs = []
        for sound in new_sounds:
            new_envs.append(sound.envelope())
            new_spectrums.append(sound.spectrum())
            new_punches.append(sound.punch())
            new_melspecs.append(sound.melspectrogram())
        for dat in (new_envs, new_melspecs, new_punches):
            dat = np.array(dat)
        new_data = [np.array(fresh) for fresh in [new_envs, new_spectrums, new_punches]]
        self.new_data = new_data
        new_melspecs = np.array(new_melspecs)
        print(new_melspecs)
        self.PCAaay(new_melspecs)
        with open(os.getcwd() + '\\HnKdata.pickle', 'rb') as f:
            data = pickle.load(f)
            f.close()
        with open(os.getcwd() + '\\HnKsounds.pickle', 'rb') as f:
            old_sounds = pickle.load(f)
            f.close()
        self.sounds = old_sounds + new_sounds
        # print(new_data.shape)
        # print(data.shape)
        for i, dat in enumerate(data):
            data[i] = np.concatenate([data[i], new_data[i]])
        self.data = data
        with open(os.getcwd() + '\\HnK_melspec.pickle', 'rb') as g:
            old_melspec = pickle.load(g)
            g.close()
        all_specs = np.concatenate([old_melspec, new_melspecs])
        self.PCAaay(all_specs)
        with open(os.getcwd()+'\\HnK_melspec.pickle','wb') as f1:
            pickle.dump(self.melspecs,f1,pickle.HIGHEST_PROTOCOL)
            f1.close()
        with open(os.getcwd()+'\\HnKdata.pickle','wb') as f2:
            pickle.dump(self.data,f2,pickle.HIGHEST_PROTOCOL)
            f2.close()
        with open(os.getcwd()+'\\HnKfile_list.dat','w') as f3:
            for file in true_names:
                f3.write(file+'\n')
            f3.close()
        with open(os.getcwd()+'\\HnKsounds.pickle','wb') as f4:
            # print('should be writing...')
            pickle.dump(self.sounds,f4,pickle.HIGHEST_PROTOCOL)
            f4.close()
        self.files=true_names
        self.indicies = np.arange(len(self.files))
    def refresh_firsttime(self):
        print('first time?')
        new_names = np.array([])
        for opendir in self.opendirs:
            new_names= np.concatenate([get_waves(opendir),new_names])
        print('making auds')
        self.sounds = []
        true_names = []
        for name in new_names:
            try:
                self.sounds.append(hk.Soundo(name))
                true_names.append(name)
            except Error:
                pass
        '''data format:
            [Envelope,FFT,Punch,melspec]
            PCA is pretty fast, so it's no big deal to run it again.
        '''
        new_names = true_names
        new_envs = []
        new_spectrums = []
        new_punches = []
        new_specs = []
        for sound in self.sounds:
            new_envs.append(sound.envelope())
            new_spectrums.append(sound.spectrum())
            new_punches.append(sound.punch())
            new_specs.append(sound.melspectrogram())
        for dat in (new_envs,new_spectrums,new_punches):
            dat = np.array(dat)
        new_data = [np.array(fresh) for fresh in [new_envs,new_spectrums,new_punches]]
        self.new_data = new_data
        new_specs = np.array(new_specs)
        self.data = new_data
        self.PCAaay(new_specs)
        print('yes indeed')
        with open(os.getcwd()+'\\HnK_melspec.pickle','wb') as f1:
            pickle.dump(self.melspecs,f1,pickle.HIGHEST_PROTOCOL)
            f1.close()
        with open(os.getcwd()+'\\HnKdata.pickle','wb') as f2:
            pickle.dump(self.data,f2,pickle.HIGHEST_PROTOCOL)
            f2.close()
        with open(os.getcwd()+'\\HnKfile_list.dat','w') as f3:
            for file in new_names:
                f3.write(file+'\n')
            f3.close()
        with open(os.getcwd()+'\\HnKsounds.pickle','wb') as f4:
            # print('should be writing...')
            pickle.dump(self.sounds,f4,pickle.HIGHEST_PROTOCOL)
            f4.close()
        self.files=new_names
        self.indicies = np.arange(len(self.files))
    def PCAaay(self,specs):
        self.melspecs = specs
        n_comp = 3
        im_dim=64
        train_data = specs.reshape(specs.shape[0],-1)
        # self.print_to_label(train_data.shape)
        sk_pca = FastICA(n_components = n_comp)
        sk_pca_data = sk_pca.fit_transform(train_data.T).T
        sk_pca_data = sk_pca_data.reshape(n_comp,im_dim,im_dim)
        comps = sk_pca.mixing_.T
        self.PCAcomps = comps.T
    def TeeSnee(self):
        # data = np.reshape(self.melspecs,(self.melspecs.shape[0],(64**2)))
        data = np.hstack([self.data[0],self.data[1],self.data[2]])
        model = TSNE(learning_rate=100, verbose=2)
        # dom = np.linspace(-1 * np.pi, 0, data.shape[1])
        # weight = (np.cos(dom) + 1) ** 2
        # weight = np.tile(weight, (data.shape[0], data.shape[1]))
        # data = data*weight
        transformed = model.fit_transform(data)
        # xs = transformed[:, 0]
        # ys = transformed[:, 1]
        return(transformed)
    def get_points(self,nodes):
        '''
        nodes = list of int indices of nodes
        
        master_datas[x][y] : x = data type, y = node

        returns = list of x,y points
        '''
        master_datas = [[1 - (self.data[y] - self.data[y][x])**2 for x in nodes] for y in range(3)]     
        means = np.array([[np.mean(master_datas[x][y],axis=1) for y in range(len(nodes))] for x in range(3)])
        master_pcs = np.array([1 - (self.PCAcomps - self.PCAcomps[x])**2 for x in nodes])
        mean_pcs = np.mean(master_pcs,axis=2)
        mean_pcs = np.interp(mean_pcs,(mean_pcs.min(),mean_pcs.max()),(-4,4))
        
        weights = np.mean(means,axis = 0)
        weights = (weights - 1.0)
        weights = np.interp(weights,(weights.min(),1),(-4,4))
        # print(mean_pcs.shape,weights.shape)
        
        theta = 2*np.pi / len(nodes)
        # print(mean_pcs.shape,weights.shape)
        truweights= (weights + mean_pcs) / 2
        angles = np.array([i * theta for i in np.arange(len(nodes))])
        nodal_vectors = np.array([[np.cos(angle),np.sin(angle)] for angle in angles])
        pos = np.dot(truweights.T,nodal_vectors)
        pos = np.exp(pos) / (1+np.exp(pos)) 
        return(pos)



lastClicked = []
nodelst = []
nodeinds = []


class Windy:
    def __init__(self):
        self.app = QtGui.QApplication([])
        self.icon = QtGui.QIcon(os.getcwd()+'\\dutch oven.png')
        self.startSize=7
        # self.app.setWindowIcon(self.icon)
        self.Statustext = QtGui.QLabel('Status stuff appears here')
        self.Initialize()

    
        

    def Initialize(self):
        if os.path.exists(os.getcwd() + '\\HnK_paths.dat'):
            openwindow = QtGui.QWidget()
            openwindow.setWindowIcon(self.icon)
            openwindow.resize(QtCore.QSize(300,300))
            openwindow.setWindowTitle('Hugs\'n\'Kisses')
            Initialtext = QtGui.QLabel('Loading Data...')
            Initialtext.setText('Loading Data...')
            image = QtGui.QLabel()
            pixmap = QtGui.QPixmap("C:\\Users\\Owner\\Pictures\\metal sponge.jpg")
            image.setPixmap(pixmap)
            layout = QtGui.QGridLayout()
            layout.addWidget(Initialtext)
            layout.addWidget(image)
            openwindow.setLayout(layout)
            openwindow.show()
            
            with open(os.getcwd()+'\\HnK_paths.dat','r') as f:
                self.paths = [path[:-1] for path in f.readlines()]
            self.backend = data_format(self.paths,Initialtext)
            self.backend.check_newbies()
            self.update_graph()
            # self.ploot.enableAutoRange()
            openwindow.close()
            self.MainWindow()


        else:
            self.path_maker(first=True)
    def update_graph(self):
        num_pts = len(self.backend.files)
        Xdim = ceil(np.sqrt(num_pts))
        Xarray = np.tile(np.arange(Xdim),Xdim).astype(float)
        Yarray = np.transpose(Xarray.reshape(Xdim,Xdim)).ravel().astype(float)
        self.pointset = np.array([Yarray[:num_pts],Xarray[:num_pts]])
        self.graph=pg.PlotWidget(enableMouse='False')
        # self.graph.setMouseEnabled(x=False,y=False)
        # self.graph.enableAutoRange()
        self.ploot = self.graph.plot(self.pointset[0],self.pointset[1],pen=None,symbol='o',clickable=True,data = self.backend.indicies,symbolSize=self.startSize)
        viewbox = self.ploot.getViewBox()
        # viewbox.setMouseEnabled(False,False)
        # print(self.ploot.getViewBox())

    def clicked(self,plot, points):
        # print(plot)
        # print(points)
        global lastClicked
        global nodelst
        for p in lastClicked:
            if p in nodelst:
        	    pass
            else:
                p.resetPen()
                p.setSize(self.startSize)
                # print(p._data[6]) #that's the index!!
        points_index = points[0]._data[6]
        points[0].setPen('b', width=2)
        points[0].setSize(10)
        if points[0] in lastClicked:
            nodelst.append(points[0])
            self.listw.insertItem(0,self.backend.files[points_index])
            points[0].setPen('r',width=2)
            nodeinds.append(points_index)
        # print(self.pointset.T)
        # yes = np.where(self.pointset.T == points_data)

        # print("clicked points", points[0]._data[0],points[0]._data[1],)
        self.backend.print_to_label(self.backend.files[points_index])
        self.backend.sounds[points_index].play()
        lastClicked = [points[0]]
    
    def define_node(self):
    	global lastClicked
    	global nodelst
    	global nodeinds
    	for p in lastClicked:
    		p.setPen('r',width=2)
    		points_index = p._data[6]
    	# points_data = [lastClicked[0]._data[0],lastClicked[0]._data[1]]
    	# yes = np.where(self.pointset == points_data)
    	self.listw.insertItem(0,self.backend.files[points_index])
    	nodelst.append(lastClicked[0])
    	nodeinds.append(points_index)
    def refresh_graph(self):
        global nodeinds
    	# self.pointset = self.backend.get_points(nodeinds)
        self.pointset = self.backend.TeeSnee()
        self.ploot.setData(self.pointset)
        self.graph.autoRange()
        self.listw.clear()
        self.autobtn.setEnabled(True)
    def Auto_Spread(self):
        '''
        finds points nearest to centroid of pointset,
        then repeats sound graph drawing.
        Uses same # nodes as original choices.
        '''
        global nodeinds
        centroid = np.mean(self.pointset,axis=0)
        sorter = self.pointset - centroid
        distances = [(item[0]**2 + item[1]**2) for item in sorter]
        idx = np.argpartition(distances,len(nodeinds))
        
        nodeinds = list(np.unique([self.ploot.scatter.data[ind][6] for ind in idx[:len(nodeinds)]]))
        print(nodeinds)
        # print(self.ploot.scatter.data)
        self.refresh_graph()
    def dragon(self):
        print('hey')
    def MainWindow(self):
        
         
        
        w = QtGui.QWidget()
        w.setWindowTitle('Hugs\'n\'Kisses')
        w.resize(QtCore.QSize(1200,600))
        w.setWindowIcon(self.icon)
        self.Statustext = QtGui.QLabel('')
        self.Statustext.setFixedSize(500,15)
        self.backend.set_label(self.Statustext)
        pathbtn = QtGui.QPushButton('Change Paths')
        nodebtn = QtGui.QPushButton('Select')
        refbtn = QtGui.QPushButton('Go!')
        self.autobtn = QtGui.QPushButton('Auto Spread')
        self.listw = QtGui.QListWidget()
        self.Statustext = QtGui.QLabel('')
        self.Statustext.setFixedSize(500,15)
        self.backend.set_label(self.Statustext)
        # print(self.graph.sceneObj.lastHoverEvent)
        self.ploot.sigPointsClicked.connect(self.clicked)
        ## Create a grid layout to manage the widgets size and position
        layout = QtGui.QGridLayout()
        w.setLayout(layout)
        pathbtn.clicked.connect(lambda: self.path_maker())
        nodebtn.clicked.connect(lambda: self.define_node())
        refbtn.clicked.connect(lambda: self.refresh_graph())
        self.autobtn.clicked.connect(lambda: self.Auto_Spread())
        self.autobtn.setEnabled(False)
        # btn.clicked.connect(lambda: print(ploot.getData()))
        ## Add widgets to the layout in their proper positions
        layout.addWidget(pathbtn, 0, 0)   # pathbutton goes in upper-left
        layout.addWidget(nodebtn,0,1) #nodebutton goes below
        layout.addWidget(refbtn,0,2)
        layout.addWidget(self.autobtn,0,3)
        layout.addWidget(self.Statustext, 1, 0,1,4)   # statustext goes in middle-left
        layout.addWidget(self.listw, 2, 0,1,4)  # list widget goes in bottom-left
        layout.addWidget(self.graph, 0, 4, 3, 1)  # plot goes on right side, spanning 3 row
        # self.graph.sigClicked.connect(clicked)
        self.window = w
        self.window.show()
        ## Display the widget as a new window
    def exe(self):
        self.window.show()
        self.app.exec_()
    def path_maker(self,first=False):
        w = QtGui.QWidget()
        w.setWindowIcon(self.icon)
        addbtn = QtGui.QPushButton('Add')
        rembtn = QtGui.QPushButton('Remove')
        retbtn = QtGui.QPushButton('Donezo')
        self.Statustext = QtGui.QLabel('Status Info Will Appear Here')
        pathbox = QtGui.QListWidget()
        layout = QtGui.QGridLayout()
        line_edit = QtGui.QLineEdit()
        w.setLayout(layout)
        w.resize(QtCore.QSize(546,243))
        layout.addWidget(addbtn,0,1)
        layout.addWidget(rembtn,1,1)
        layout.addWidget(self.Statustext,1,0)
        layout.addWidget(line_edit,0,0)
        layout.addWidget(pathbox,2,0,5,1)
        layout.addWidget(retbtn,2,1)
        if first==True:
            # print('hey')
            w.setWindowTitle('Baby\'s First Paths :\')')
            first = True
            pass
        else:
            first=False
            w.setWindowTitle('Edit Paths')
            with open(os.getcwd()+'\\HnK_paths.dat','r') as f:
                old_paths = [path[:-1] for path in f.readlines()]
                pathbox.addItems(old_paths)
                f.close()
        addbtn.clicked.connect(lambda: pathbox.insertItem(0,line_edit.text()))
        rembtn.clicked.connect(lambda: pathbox.takeItem(pathbox.currentRow()))
        
        def update_path_list():
            paths = []
            with open(os.getcwd()+'\\HnK_paths.dat','w') as f:
                for i in range(pathbox.count()):
                    if pathbox.item(i) == '':
                        pass
                    else:  
                        paths.append(pathbox.item(i).text())
                        f.write('{}\n'.format(pathbox.item(i).text()))
                f.close()
            self.backend = data_format(paths,self.Statustext)
            self.backend.refresh_firsttime()

            self.update_graph()
            self.MainWindow()
        retbtn.clicked.connect(lambda: update_path_list())
        self.window = w
        self.window.show()


app = Windy()
app.exe()

# pls = data_format([opendir3,opendir2])

# pls.check_newbies()

# pos = pls.get_points([3,4,5,6])

