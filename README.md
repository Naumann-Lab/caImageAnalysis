# caImageAnalysis
<img align = "right" width = "240" src="mascot.png ">

### Introduction

Calcium imaging functional analysis

<br>

The default class is a BaseFish from fishy.py. We offer a process.py for various analyses.
The standard implementation runs motion correction via caiman and then source extraction via suite2p, 
and most of the more advanced classes assume a suite2p folder with contained sources.


<br>


<br>
<br>
We leverage a folder structure that contains all experiment items

### Folder Structure

<img align = "left" width = "500" src="resources\folder_overview.PNG ">

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

### Installation
This assumes you are using Anaconda and Python 3:

1. Install [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge)


2. Create a new environment and install [Caiman](https://caiman.readthedocs.io/en/master/Installation.html#installing-caiman). Note: helpful information on caiman can be found [here](https://github.com/EricThomson/CCN_caiman_mesmerize_workshop_2023)
    
    `mamba create -n caiman -c conda-forge caiman`


3. Activate the envs (note: python should be 3.7x)

   `mamba activate caiman`


4. Install [suite2p](https://github.com/MouseLand/suite2p)

   ` python -m pip install suite2p`


5. Edit suite2p documentation to fit with scipy and numpy versions compatible with caiman

    \Lib\site-packages\suite2p\detection\sparsedetect.py 

   Remove the 'keepdims' arg from line 256


6. Install more packages

   ` python -m pip install tables nptdms pyarrow`


7. Optional: Download [mimic_alpha](https://github.com/montefra/mimic_alpha) into your new caiman envs - this package converts a list of RGB color that mimic a RGBA on a given background. This is useful for plotting.




<br>

### Structure


core: motion correction, source extraction, calculate factors <br> 

utils: ideally things used multiple places, converts timestamps, stimuli, etc into dataframes <br> 

visualize: various visualization of response-classes, barcodes, neuron functions, etc <br> 



<br><br><br>
Issues:
sometimes processing multiple fish at a time gives permission errors: unclear reproducibility, just refresh and give it another go