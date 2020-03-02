import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import tkinter as tk
from tkinter import ttk
# from tkinter import Canvas
from tkinter.filedialog import askdirectory,askopenfilename
from tkinter import StringVar,IntVar
import os
import hugsnkisses.Aud as hk
import mplcursors
from PIL import Image,ImageTk

opendir ="C:/Users/Owner/Documents/Audio Processing Stuff/WAAay Bigger Drumset/"

thinkin_dir = "C:/Users/Owner/Pictures/thinkin/"

thinkins = [thinkin_dir+thank for thank in os.listdir(thinkin_dir)]

Times=('Times', 12)

class TheApp(tk.Tk):
    
    def __init__(self,*args,**kwargs):
        
        tk.Tk.__init__(self,*args,**kwargs)
        # tk.Tk.iconbitmap(self,os.getcwd()+'\GEIcon.ico')
        tk.Tk.wm_title(self,"Hugs 'n' Kisses")
        container = tk.Frame(self)
        

        
        container.grid()
        
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)
        
        self.frames = {}
        
        frame=Initial_picker(container,self)
            
        self.frames[Initial_picker] = frame
            
        frame.grid(row=0,column=0,pady=15)

        w = frame.winfo_screenwidth()
        h = frame.winfo_screenheight()
        # self.overrideredirect(True)
        self.geometry('%dx%d+0+0' % (w, h))
        
        self.show_frame(Initial_picker)

    def show_frame(self,cont):
        frame = self.frames[cont]
        frame.tkraise()
    # def show_frame_and_pass(self,cont,pickle_path):
    #     frame=ParamsPage(self,cont,pickle_path)
    #     self.pickle_path=pickle_path
    #     frame.grid(row=1,column=0,pady=15)
    #     self.frames[cont]=frame
    #     frame.tkraise()
#    def Range_Page(self,cont,Content):
#        frame=RangePage(self,cont,Content)
#        self.Content=Content
#        frame.grid(row=1,column=1,pady=15)
#        self.frames[cont]=frame
#        frame.tkraise()

old=0
class Initial_picker(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        # self.imager = Canvas(self,width=500,height=500)
        self.forreal()
        # self.imager.grid(row=0,column=0)
    def forreal(self):
        frontside=StringVar()
        backside=StringVar()
        front_skiprows=IntVar()
        back_skiprows=IntVar()
        Frontside_dir= tk.Label(self,text='Pick up the headphones to hear things\nDouble Click to Pick a Sound\n(At least 2,no more than 7)\nThen click "Process!"\nProcessing takes a sec.\nPlease be patient.\nPlease limit sequences to 7 sounds\n-Management',font=Times)
        Frontside_dir.grid(row=0,column=1,padx=5,pady=7)
        fig = Figure(figsize=(10,6),dpi=100)
        ax = fig.add_subplot(111)
        ax.set_facecolor((0.5,0.5,0.5,1))
        ax.set_title('This is start of the program.')
        canvas = FigureCanvasTkAgg(fig,self)
        canvas.get_tk_widget().grid(row=0,column=0,rowspan=7)
        # thinkin = tk.PhotoImage(file=np.random.choice(thinkins))
        # imager.config(image = thinkin)
        # imager = tk.Label(self,image = thinkin)
        # imager.grid(row=0,column=0,rowspan=6)
        node_box=tk.Listbox(self,font=Times)
        node_box.grid(row=3,column=1)
        name_box = tk.Entry(self,font=Times,width=30)
        name_box.grid(row=2,column=1,padx=10)
        names = os.listdir(opendir)
        sounds = [hk.Aud(opendir + file) for file in names]
        def Initialize_graph(fig=fig,ax=ax):
            ax.clear()
            x=np.arange(25)
            y=np.arange(26)
            Xx,Yy = np.meshgrid(x,y)
            X,Y=np.ravel(Xx),np.ravel(Yy)
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
            bbox_props = dict(boxstyle="sawtooth,pad=0.3", ec="black", lw=2)
            sc = ax.scatter(np.ravel(X),np.ravel(Y),s = 40,c='black')
            annots = []
            node_box.delete(0,'end')
            def hover(event,self=self):
                global old
                global nodes
                name_box.delete(0,'end')
                if event.inaxes == ax:
                    cont, ind = sc.contains(event)
                    if cont:
                        ii=ind["ind"][0]
                        sounds[ii].play()
                        name_box.insert(0,names[ii])
                        fig.canvas.draw_idle()
                        if ii==old:
                            if names[ii] in node_box.get(0,'end'):
                                pass
                            else:
                                node_box.insert(0,names[ii])
                        old = ii
                    else:
                        pass
            canvas.draw()
            self.cid=canvas.mpl_connect('button_press_event',hover)
        Initialize_graph()
        def update_plot(self=self):
            canvas.mpl_disconnect(self.cid)
            ax.clear()
            nodes = node_box.get(0,'end')
            # thinkin = Image.open(np.random.choice(thinkins)).convert('RGB')
            # ax.imshow(thinkin)
            # print('')
            plotter,vecs = hk.Spiderer(nodes,sounds,names)
            # print('end')
            # imager.config(image='')
            # ax.clear()
            # node_box.delete(0,'end')
            sc = ax.scatter(plotter['x'].values,plotter['y'],s=25,c='black')
            ax.set_title('some sequences are invalid.')
            def clicker(event):
                global old
                name_box.delete(0,'end')
                if event.inaxes == ax:
                    cont, ind = sc.contains(event)
                    if cont:
                        ii=ind["ind"][0]
                        sounds[ii].play()
                        name_box.insert(0,names[ii])
                        fig.canvas.draw_idle()
                        old = ii
                        if ii==old:
                            if names[ii] in node_box.get(0,'end'):
                                pass
                            else:
                                node_box.insert(0,names[ii])
                    else:
                        pass
            for i,vec in enumerate(vecs):
                ax.scatter(vec[0],vec[1],c='red')
                # ax.annotate(nodes[i],vec)
            canvas.draw()
            canvas.mpl_connect('button_press_event',clicker)
            node_box.delete(0,'end')
        Choose_button = ttk.Button(self,text='Process!',command=lambda:update_plot())
        Choose_button.grid(row=1,column=1)
        clear_button = ttk.Button(self,text='Delete Last Pick',command=lambda:node_box.delete(0))
        clear_button.grid(row=4,column=1)
        reset_button = ttk.Button(self,text='Restart',command = lambda:Initialize_graph())
        reset_button.grid(row=5,column=1)
        sequencer_button = ttk.Button(self,text='Play your list', command = lambda: hk.sequencer([hk.Aud(opendir+hit) for hit in node_box.get(0,'end')]))
        sequencer_button.grid(row = 6,column=1)
         
         
app=TheApp()
app.mainloop()