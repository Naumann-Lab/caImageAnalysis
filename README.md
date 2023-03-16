# caImageAnalysis
<img align = "right" width = "240" src="mascot.png ">

### Introduction

Calcium imaging functional analysis


Pipeline leverages caiman for image motion correction and suite2p for source extraction (caiman can also be used for this)

<br>

### Installation
This assumes you are using Anaconda and Python 3:

1. Install [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge)


2. Create a new environment and install [Caiman](https://caiman.readthedocs.io/en/master/Installation.html#installing-caiman)
    
    `mamba create -n caiman -c conda-forge caiman`


3. Activate the envs (note: python should be 3.7x)

   `mamba activate caiman`


4. Install [suite2p](https://github.com/MouseLand/suite2p)

   ` python -m pip install suite2p`


5. Edit suite2p documentation to fit with scipy and numpy versions compatible with caiman

    \Lib\site-packages\suite2p\detection\sparsedetect.py 

   Remove the 'keepdims' arg from line 256


6. Install more packages

   ` python -m pip install tables nptdms`


Helpful information on caiman can be found [here](https://github.com/EricThomson/CCN_caiman_mesmerize_workshop_2023)

<br>

### Structure


core: motion correction, source extraction, calculate factors <br> 

utils: ideally things used multiple places, converts timestamps, stimuli, etc into dataframes <br> 

visualize: various visualization of response-classes, barcodes, neuron functions, etc <br> 

<br><br><br>
Issues:
sometimes processing multiple fish at a time gives permission errors: unclear reproducibility, just refresh and give it another go