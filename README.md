# hugsnkisses-repo
Hey all.

Welcome to hugsnkisses, a relatively simple means to sort drum samples against each other, without actually having to go in and do the dirty work of sorting samples yourself.

##### Dependencies
1. librosa, simpleaudio, soundfile, and scipy.signal for audio processing modules
2. numpy for data wrangling
3. PyQt5/pyqtgraph for GUI,plotting, and plot callbacks.

###Instructions
When you run QtPresentation.py for the first time, the "Baby's First paths" window appears. 
Extract the Way Bigger Drumset and copy-paste its path into the textbox, then hit Donezo.
It'll scan the folder (and its sub-folders) for wav files (other sound files not currently supported) and generate all the relevant data. It'll store this data in the files HnKsounds.pickle, HnKdata.pickle, and HnK_melspec.pickle.
If it doesn't detect ANY ONE of these files on startup, it'll start from scanning the paths in HnK_paths.dat.

The initial graph is a grid of all the samples. To run the t-SNE algorithm, hit the Go button. It'll process for a bit, then display the new arrangement of your sounds.  

## Operation 
The old algorithm worked like this:

The user picks three or more samples from the initial grid of samples. The other samples are compared against these, and the final graph is drawn.

The final graph first puts the `N` master samples evenly spaced on a unit circle. For each of the `n` rest-of-the samples, a vector is computed via a weighted sum of the categories. These are the PCA component coefficients and the sum of log-error of amplitude and frequency spectrum. 
For each of the `n` samples, `N` weights are computed corresponding to the master samples. This vector gets multiplied by the `N` basis vectors, and the position of the sample is thus computed.

The new algorithm is an implementation of t-SNE, which is better for this application. In the future, I'd like to combine the old and the new to get a customizable t-SNE hybrid.  
## QtPresentation.py

I've changed from tkinter GUI to PyQT5. This enables use of the pyqtgraph module for faster plotting, and all-around neatness.

The graphs are generated in Aud.py and displayed in a big frame, buttons are on the left.

## Aud.py

Aud.py is the backend for QtPresentation.py.

On loading, each sound gets loaded into its own instance of the Aud class. This contains all the relevant FFT methods, as well as the `play` method for playback when the point is clicked. 

This file also contains the old functions used for plotting the sounds. I've kept them for experimentation, and sentimentality. 

####Known Issues

Some wav files simply won't load, not due to samplerate or anything else. This is an issue with simpleaudio, and it's currently unsolved. It seems to be a problem with Python itself. ``

I'm working on putting console output in the loading window to make operation more clear, and to reduce fear when "Not Responding" appears.