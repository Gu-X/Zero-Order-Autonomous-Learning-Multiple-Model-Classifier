# Copyright 2018, Plamen P. Angelov and Xiaowei Gu

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.





# This code is the Autonomous Learning Multi-Model System of Zero Order described in:
#==========================================================================================================
# P. P. Angelov and X. Gu, “Autonomous learning multi-model classifier of 0-order (ALMMo-0),” 
# in IEEE International Conference on Evolving and Autonomous Intelligent Systems, 2017, pp. 1–7.
#==========================================================================================================
# Please cite the paper above if this code helps.

# For any queries about the code, please contact Prof. Plamen P. Angelov and Mr. Xiaowei Gu 
# {p.angelov,x.gu3}@lancaster.ac.uk
# Programmed by Xiaowei Gu

import numpy
import math
import scipy 
OR=1-math.cos(math.pi/6) # Set the initial radius for each newly dadded data cloud;
# This value is changeable
#################
def Learning(data,systemPara):# Train the ALMMo-0 system
    data=numpy.array(data)
    data=data/numpy.sqrt(sum(data**2)) # Normalise the data sample with its norm
    if systemPara=={}: #Initialise the system
        systemPara['K']=1 # The current time index
        systemPara['Centre']=numpy.matrix(data)  # The centre of the first data cloud
        systemPara['Support']=numpy.array([1]) # The support of the first data cloud (number of members)
        systemPara['X']=numpy.array([1]) # The average scalar product of the members of the first data cloud
        systemPara['Centre_number']=1 # Number of existing prototypes/centres
        systemPara['Radius']=numpy.array([OR]) # The radius of the area of influence of the first data cloud
        systemPara['Global_mean']=data # Global mean
        systemPara['Global_X']=1 # Global average scalar product
    else:
        systemPara['K']=systemPara['K']+1 # Update the time instance
        systemPara['Global_mean']=(systemPara['Global_mean']*(systemPara['K']-1)+data)/systemPara['K'] # Update the global mean
        GDelta=systemPara['Global_X']-sum(systemPara['Global_mean']**2)
        CentreDensity=numpy.zeros(systemPara['Centre_number'])
        for ii in range(0,systemPara['Centre_number']):  # Calculate the global density at the prototypes of the existing data clouds
            CentreDensity[ii]=1/(1+numpy.sum(numpy.power(systemPara['Centre'][ii,:]-systemPara['Global_mean'],2))/GDelta) 
        MACD=numpy.amax(CentreDensity)
        MICD=numpy.amin(CentreDensity)
        DataDensity=1/(1+sum((data-systemPara['Global_mean'])**2)/GDelta) # Calculate the global density at the new data sample
        if (DataDensity>MACD) or (DataDensity<MICD): # Add a new data cloud/prototype to the system
            systemPara['Centre_number']=systemPara['Centre_number']+1
            systemPara['Centre']=numpy.append(systemPara['Centre'],[data],axis=0)
            systemPara['Support']=numpy.append(systemPara['Support'],[1],axis=0)
            systemPara['Radius']=numpy.append(systemPara['Radius'],[OR],axis=0)
            systemPara['X']=numpy.append(systemPara['X'],[1],axis=0)
        else:
            dist0=scipy.spatial.distance.cdist(numpy.matrix(data), systemPara['Centre'], 'euclidean') # Calculate the distances between the new data sample and the existing data clouds
            idx0=numpy.argmin(dist0,axis=1) # Find the nearest data cloud 
            dist1=numpy.power(dist0[0,idx0],2)/2
            if (dist1>systemPara['Radius'][idx0]): # Add a new data cloud/prototype to the system
                systemPara['Centre_number']=systemPara['Centre_number']+1
                systemPara['Centre']=numpy.append(systemPara['Centre'],[data],axis=0)
                systemPara['Support']=numpy.append(systemPara['Support'],[1],axis=0)
                systemPara['Radius']=numpy.append(systemPara['Radius'],[OR],axis=0)
                systemPara['X']=numpy.append(systemPara['X'],[1],axis=0)
            else: # Update the meta-parameters of the nearest data cloud
                systemPara['Support'][idx0]=systemPara['Support'][idx0]+1
                systemPara['Centre'][idx0,:]=(numpy.multiply(systemPara['Centre'][idx0,:],(systemPara['Support'][idx0]-1))+numpy.matrix(data))/systemPara['Support'][idx0]
                systemPara['Radius'][idx0]=systemPara['Radius'][idx0]/2+(systemPara['X'][idx0]-numpy.sum(numpy.power(systemPara['Centre'][idx0,:],2)))/4;
    return systemPara

#################
def Testing(data,systemPara):# Conduct the validation with the pre-trained ALMMo-0 system
    CL=systemPara['Class_number']
    data=numpy.matrix(data)
    data=data/numpy.sqrt(numpy.sum(numpy.power(data,2)))
    T=numpy.zeros(CL)
    for ii in range (0,CL):
        dist0=scipy.spatial.distance.cdist(data, systemPara[ii]['Centre'], 'euclidean')
        T[ii]=numpy.amin(dist0)
    Label=numpy.argmin(T,axis=0)
    return Label+1

#################
def ALMMo0classifier_learning(tradata,tralabel): #ALMMo-0 system for learning
    L1,W1=tradata.shape
    CL=numpy.amax(tralabel)
    tradata1=[]
    for ii in range(1,CL+1):
        tradata1.append(tradata[numpy.where(tralabel==ii)[0]])
    systemPara={}     
    for ii in range (0,CL):
        systemPara[ii]={}
        status='L'
        L3,W1=tradata1[ii].shape
        systemPara[ii]={}
        systemPara[ii]=Learning(tradata1[ii][0,:],systemPara[ii])
        for jj in range(1,L3):
            systemPara[ii]=Learning(tradata1[ii][jj,:],systemPara[ii])
    systemPara['Class_number']=CL # Number of existing classes
    return systemPara

#################
def ALMMo0classifier_testing(tesdata,systemPara): #ALMMo-0 system for learning
    L2,W1=tesdata.shape
    TestLabel=numpy.zeros(L2)
    for ii in range(0,L2):
        TestLabel[ii]=Testing(tesdata[ii,:],systemPara)
    return TestLabel
    
        


