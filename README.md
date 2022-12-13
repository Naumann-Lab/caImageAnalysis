# caImageAnalysis
<img align = "right" width = "240" src="mascot.png ">

python 3.7 caiman + suite2p environment

calcium imaging functional analysis


pipeline leverages caiman for image motion correction and suite2p for source extraction (caiman can also be used for this)


<br>


core: motion correction, source extraction, calculate factors <br> 

utils: ideally things used multiple places, converts timestamps, stimuli, etc into dataframes <br> 

visualize: various visualization of response-classes, barcodes, neuron functions, etc <br> 

<br><br><br>
Issues:
sometimes processing multiple fish at a time gives permission errors: unclear reproducibility, just refresh and give it another go