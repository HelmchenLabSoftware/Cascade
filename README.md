## Calibrated inference of spiking (work in progress)

People currently involved: Peter Rupprecht, Adrian Hoffmann

### Installation

Dependencies are:

- numpy, scipy and other typical stuff that is included in Anaconda for example
- tensorflow and keras (as a deep learning framework); this can be installed in an Anaconda environment either as CPU- or GPU-based framework; detailed instructions should be provided here

### Overview of main files

'Main_program.py' contains a standalone version of the program. Variables like the training dataset(s), the test dataset, the framerate of the test dataset, the ensemble size and the model parameters have to be adjusted manually in-line. This script would be a starting point to create something independent and automatized.

'GUI.py' contains the same functionalities, but provides a clean and simple GUI. This GUI needs to be improved and should include a lot of tooltips when hovering over the buttons or when the user tries to do something stupid.

### Overview of helper files

'helper_scripts' contains a lot of functions that are used to read data, to resample the ground truth and to fit spikes into the predicted spiking probabilities

'GT_datasets' contains all ground truth datasets as \*.mat files. They are called "mini" because the files do not contain the full ROIs, raw ephys data and other large things.

'Test_dataset' contains a test dataset, acquired in zebrafish with OGB-1 at a framerate of 7.5 Hz. Includes 21 trials, each of them with ca. 1000 neurons recorded for a bit longer than 30 sec.

### To do list

- The relative paths of the ground truth datasets in the GUI and the 'Main_program.py' are not correct: 'DONE'
- Run scripts from the repository as it is
- Z
