from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as et
from io import StringIO
import csv

###Only need to change this###
K = 93

directories = ['15', '16', '17', '20', '22', '23']
powerLevels = ['0', '65', '74', '80', '82', '85']

################


fig, ax = plt.subplots(nrows= 3, ncols= len(directories))
plt.figure(figsize=(16,28))


for dirNum in range(len(directories)):
    rootDirectory = 'C:/Users/honey/Downloads/20230720_N397a_h2bgcamp6s_elavl3rsChr_7dpf/Paralyzed Fish/photostim_rest_visstim_roi2_' + powerLevels[dirNum]+ 'power-0'+ directories[dirNum]+'/'
    if(powerLevels[dirNum] == '0'):
        rootDirectory = 'C:/Users/honey/Downloads/20230720_N397a_h2bgcamp6s_elavl3rsChr_7dpf/Paralyzed Fish/vis_stim_baseline_roi2-015/'
        photoStim = False
    else:
        photoStim = True
    #based on the last file in the root directory, we can get the base file name and generate others#
    inds = []
    for i in range(len(rootDirectory)):
        if(rootDirectory[i] == '/'):
            inds.append(i)
    pos = inds[-2]
    baseName = rootDirectory[pos+1:-1]
    #generate names of subfiles
    botFname = rootDirectory + baseName + '_Cycle00001-botData.csv'
    treeFname = rootDirectory + baseName + '.xml'
    imageFname = rootDirectory + baseName + '_Cycle00001_Ch2_000001.ome.tif'
    stimTreeFname = rootDirectory + baseName + '_Cycle00001_MarkPoints.xml'

    #get brightness over time csv data into a numpy dataframe
    botDF = pd.read_csv(botFname)
    botNP = botDF.to_numpy()

    #open the brightness over time xml file and find the regions' x and y centers(?) and their constituent points
    botTree = et.parse(treeFname)
    botRoot = botTree.getroot()
    botRegions = []
    #go through the child nodes of the tree until the nodes with coordinates are found then add them to our list of regions
    for child in botRoot:
        if(child.tag == 'Sequence'):
           for grandchild in child:
               if(grandchild.tag == 'PVBOTs'):
                  for greatgrandchild in grandchild:
                    botRegions.append(greatgrandchild.attrib)
    #take the points out of their string format and get them into a numpy array
    strIO = StringIO(botRegions[0]['points'])
    reader = csv.reader(strIO, delimiter = ',')
    numberList = None
    for row in reader:
        numberList = row
    numberList = np.float32(numberList)

    #construct an np array from the tiff stack
    #open image and create an empty array
    img = Image.open(imageFname)
    myArray = np.zeros((np.shape(img)[0], (np.shape(img)[1]), img.n_frames))
    #read each frame into the array
    for i in range(img.n_frames):
        img.seek(i)
        myArray[:,:,i] = img

    if(photoStim):
        #find the coordinates used for photostimulation
        #load the photostim xml file
        etree = et.parse(stimTreeFname)
        #find the root
        root = etree.getroot()
        myX = None
        myY = None
        #go through nodes until you get to the correct depth to find the x and y coords (this only works for one photostim)
        #FIXME
        for node in root:
            for l1 in node:
                for l2 in l1:
                    myX = np.float32(l2.attrib.get("X"))
                    myY = np.float32(l2.attrib.get("Y"))
        #translate the %-based coordinates to pixel coordinates
        xCoord = np.shape(img)[1] * myX
        yCoord = np.shape(img)[0] * myY
    else:
        xCoord = None
        yCoord = None
    #calculate a mean brightness trace
    brightnessArray = myArray.mean(axis=(0,1))
    #calculate mean and 0.4 "median" quantile for dF
    meanArray = myArray.mean(2)
    medianArray = np.quantile(myArray, (0.4), axis = 2)
    badframes = []
    if(photoStim):
        #use large changes in brightness (due to PMT shutter closure for laser) to do timing
        diffArray = np.diff(brightnessArray)
        #detects end of photostim
        # identify frames with large brightness changes
        ids = np.squeeze(np.where(diffArray > 10))
        beginIds = np.squeeze(np.where(diffArray < -10))

        # remove frames that are essentially duplicates
        i = 0
        while (i < len(ids) - 1):
            print(ids[i], ids[i + 1] - 1)
            if (ids[i] == ids[i + 1] - 1):
                ids = np.delete(ids, i + 1)
                i = i - 1
            i = i + 1
        i = 0
        while (i < len(beginIds) - 1):
            if (beginIds[i] == beginIds[i + 1] - 1):
                beginIds = np.delete(beginIds, i + 1)
                i = i - 1
            i = i + 1
        for i in range(len(beginIds)):
            for number in range(beginIds[i], ids[i]):
                badframes.append(number)

        #we did 5 stimulations and there are 93 frames between them, so grab just these frames
        mySlices = np.zeros((5,(np.shape(img)[0]), (np.shape(img)[1]),K))
        for i in range(5):
            index = ids[0][i]
            slice = myArray[:,:,index:index+K]
            mySlices[i,:,:,:] = slice
        #calculate the mean of all 93*5 frames
        meanEffect = mySlices.mean(axis=(0,3))
        #calculate the dF for the effect of photostim
    else:
        meanEffect = np.mean(myArray[:,:,50:450],axis=2)
    #take the mean of the visual stimulation and find the dF for vis stim's effect
    dFeffect = (meanEffect - medianArray) / meanArray
    visZone = myArray[:,:,-400:]
    visMean = visZone.mean(axis=2)
    dFvis = (visMean-medianArray)/meanArray

    #plot the dF vs baseline for photostimulation
    ax[0,dirNum].imshow(dFeffect)
    #ax[0,dirNum].scatter(xCoord,yCoord, color = 'r', marker='+')


    #plot the dF vs baseline for visual stimulation
    ax[1,dirNum].imshow(dFvis)
    #ax[1,dirNum].scatter(xCoord,yCoord, color = 'r', marker='+')

    ax[2, dirNum].plot(range(img.n_frames), brightnessArray)

    ax[2,dirNum].set_xlabel(powerLevels[dirNum]+' AU')

    if not photoStim:
        ax[1,dirNum].set_ylabel("Visual Stimulation")
        ax[0, dirNum].set_ylabel("Photostimulation")

plt.show()
