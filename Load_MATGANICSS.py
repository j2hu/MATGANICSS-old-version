#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os            # linux shell
import re            #^[]?%
import pymatgen as mg  #materials project
import pymatgen.analysis.diffraction as anadi
import pymatgen.analysis.diffraction.xrd as xrd
import numpy as np
import glob          #random
import matplotlib.pyplot as plt
import torch         #machine learning
import torch.nn as nn
from torch.autograd import Variable


# In[ ]:


torch.set_default_dtype(torch.float64)

torch.set_printoptions(precision=16)

patt_xrd = xrd.XRDCalculator('CuKa')

train_path='/home/mii/trainCSSO/O5/train/'

test_path='/home/mii/trainCSSO/O5/test/'


# In[ ]:


global sample_num, rmat_num, atoms_num
sample_num=1
rmat_num=28  #row nums of the matrix for the input of CNN 
atoms_num=24

global move_num,extend_num
extend_num=100

LR_D=0.001  #learning rate
LR_G=0.001

#get_energy(train_path+'0000')
move_num=-115


# In[ ]:


def random_xxpsk(file_path):
    folder=np.random.choice(glob.glob(file_path +"*"))
    return folder


# In[ ]:


def get_energy(folder):
    energy_string=os.popen('grep TOTEN '+folder+'/OUTCAR | tail -1').read().split(' ')[-2]
    energy=np.float64(float(energy_string))
    return energy

def linear_transform(energy):
    global extend_num, move_num
    energy_transform=(energy-move_num)*extend_num
    return energy_transform
def inverse_transform(energy_transform):
    global extend_num, move_num
    energy=energy_transform/extend_num+move_num
    return energy
def get_energy_per_atom(energy):
    energy_per_atom=energy/atoms_num
    return energy_per_atom


# In[ ]:


def tomgStructure(folder):
    POSfile=folder+'/CONTCAR'
    R_mgS=mg.Structure.from_file(POSfile)
    return R_mgS
    #print(POSfile)
    
def get_xrdmat4(mgStructure):
    global rmat_num
    xrd_data4 =patt_xrd.get_pattern(mgStructure)
    i_column = rmat_num
    xxx=[]
    yyy=[]
    mat4=[]
    xrd_i=len(xrd_data4)
    for i in range(xrd_i):
        if xrd_data4.y[i]>1 and xrd_data4.y[i] < 20:
            xxx.append(xrd_data4.x[i])
            yyy.append(xrd_data4.y[i])
    mat4.append(np.asarray(xxx))
    mat4.append(np.asarray(yyy))
    mat4=np.asarray(mat4)
    
    xrd_x=[]
    xrd_y=[]
    xrd_mat4=[]
    xrow=len(mat4[0])
    
    if xrow < i_column:
        for i in mat4[0]:
            xrd_x.append(i)
        for j in mat4[1]:
            xrd_y.append(j)
        for i in range(0,i_column-xrow):
            xrd_x.append(0)
            xrd_y.append(0)
        xrd_x=np.asarray(xrd_x)
        xrd_y=np.asarray(xrd_y)
    if xrow > i_column:
        xrd_x=mat4[0][:i_column]
        xrd_y=mat4[1][:i_column]
    if xrow == i_column:
        xrd_x= mat4[0]
        xrd_y= mat4[1]
        
    xrd_x=np.sin(np.dot(1/180*np.pi,xrd_x))
    xrd_y=np.dot(1/100,xrd_y)
    xrd_mat4.append(xrd_x)
    xrd_mat4.append(xrd_y)
    xrd_mat4=np.array(xrd_mat4)
    return xrd_mat4

def GANs_Gmat(Random_Structure):
    global rmat_num
    RS_xrdmat = get_xrdmat4(Random_Structure)
    multimat3_RS =  np.zeros((rmat_num,rmat_num),dtype='float32')
    multimat3_RS = np.asarray(np.sqrt(np.dot(RS_xrdmat.T, RS_xrdmat)))
    return multimat3_RS


# In[ ]:


class GcssNet(nn.Module):
    def __init__(self, input_size=(sample_num,28,28)):
        super(GcssNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(#(3,28,28)
                in_channels=sample_num,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),#->(32,28,28)
            nn.ReLU(),#->(32,28,28)
            nn.MaxPool2d(kernel_size=2),
        )#->(#->(32,14,14))
        self.conv2=nn.Sequential(#->(32,14,14))
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),#->(64,14,14)
            nn.ReLU(),#->(64,14,14)
            nn.MaxPool2d(kernel_size=2),#->(64,7,7)
        )
        self.out=nn.Sequential(
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,sample_num),            
        )
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x) #batch(64,7,7)
        x=x.view(x.size(0),-1) #(batch, 64*7*7)
        output=torch.unsqueeze(self.out(x),dim=0)
        return output

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.Dlstm=nn.LSTM(
            input_size=sample_num,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out=nn.Sequential(
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid(),
        )
        #nn.Linear(32,1)
        #nn.Relu
        #nn.Linear
        #nn.Sigmoid
        
    def forward(self,x):
        D_out,(h_n,h_c)=self.Dlstm(x,None)
        out = self.out(D_out[:,-1,:]) #(batch,time step,input)   
        return out


# In[ ]:


G1=GcssNet()
D1=DNet()


# In[ ]:




