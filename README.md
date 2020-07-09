# hugsnkisses-repo
Hey all.

Welcome to hugsnkisses, a relatively simple means to sort drum samples against each other, without actually having to go in and do the dirty work of sorting samples yourself.

##### Dependencies
1. librosa, simpleaudio, soundfile, and scipy.signal for audio processing modules
2. pandas for data wrangling
3. pyqtgraph for plotting and plot callbacks.

## Operation 
The user picks three or more samples from the initial grid of samples. The other samples are compared against these, and the final graph is drawn.

The final graph first puts the `N` master samples evenly spaced on a unit circle. For each of the `n` rest-of-the samples, a vector is computed via a weighted sum of the categories. These are the PCA component coefficients and the sum of log-error of amplitude and frequency spectrum. 
For each of the `n` samples, `N` weights are computed corresponding to the master samples. This vector gets multiplied by the `N` basis vectors, and the position of the sample is thus computed.

## Presentation.py

\The GUI is currently built on tkinter, but I plan to change to PyQT5 in a future version.
I've changed from tkinter GUI to PyQT5. This enables use of the pyqtgraph module for faster plotting.

The graphs are generated in Aud.py and displayed in a big frame, buttons are on the right.

## Aud.py

Aud.py is the backend for Presentation.py.

The samples are sorted based on simple PCA of the mel-spectrograms, as well as a direct sum of the log-error of the amplitude curves and fourier transforms.

All the relevant numbers are put in a big ol' dataframe, and the vector operations are then performed to draw the final graph.

##### Spiderer

This function does the bulk of computation, and draws the graph. I'm gonna vectorize it soon, as it currently uses for-loops for the whole shebang. This is not very cash money.

##### Sound_graph

Takes a plot and a list of Aud objects, and connects the Aud.play() to clicking the corresponding point on the graph. Unfortunately, plot callbacks get picked up by garbage collection.
