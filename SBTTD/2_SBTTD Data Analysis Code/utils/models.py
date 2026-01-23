import tecplot as tp
import numpy as np
import os

examples_dir = tp.session.tecplot_examples_directory()
df = os.path.join(examples_dir, 'OneraM6wing', 'OneraM6_SU2_RANS.plt')
outfile=os.path.join('out.plt')
ds = tp.data.load_tecplot(df)

listOf2DZnes=[]

for Zne in ds.zones():
    if Zne.rank==2:
        listOf2DZnes.append(Zne.index)

#Below are specific to the dataset
tp.macro.execute_extended_command('CFDAnalyzer4',
    "SetFieldVariables ConvectionVarsAreMomentum='T' UVar=5 VVar=6 WVar=7 ID1='Density' Variable1=4 ID2='NotUsed' Variable2=0")
tp.macro.execute_extended_command('CFDAnalyzer4',
	  'Calculate Function=\'GRIDKUNITNORMAL\' Normalization=\'None\' ValueLocation=\'Nodal\' CalculateOnDemand=\'F\' UseMorePointsForFEGradientCalculations=\'F\'')
tp.macro.execute_extended_command('CFDAnalyzer4',
	  'Calculate Function=\'VELOCITYMAG\' Normalization=\'None\' ValueLocation=\'Nodal\' CalculateOnDemand=\'F\' UseMorePointsForFEGradientCalculations=\'F\'')
v0=285#Reference velocity magnitude
iVmag=ds.variable('Velocity Magnitude').index
      
#Define the identification of the BL here. It should return 3 booleans:
# In the BL, out of the BL, on the BL (=store result)
def BL95(xp,yp,zp,v0,iVmag):
    try:
        v1=tp.data.query.probe_at_position(xp,yp,zp)
        v=v1.data[iVmag]
        out=False
    except:
        outBL=False
        inBL=False
        bl=False
        v=v0
        out=True
    vr=v/v0
    #print("Vr={}".format(vr))
    if 0.96<vr:
        outBL=True
        inBL=False
        bl=False
    elif vr<0.94:    
        outBL=False
        inBL=True
        bl=False
    else:
        outBL=False
        inBL=False
        bl=True
        #print("in da BL")
    return(inBL,outBL,bl,out)
        
        
      
      
tp.data.operate.execute_equation('{BL_Thickness}=0')
for iZ in listOf2DZnes:
    Zne=ds.zone(iZ)
    resultZne=ds.copy_zones(iZ)
    resultZne=resultZne[0]
    x=Zne.values('x')
    y=Zne.values('y')
    z=Zne.values('z')
    xn=Zne.values('X Grid K Unit Normal')
    yn=Zne.values('Y Grid K Unit Normal')
    zn=Zne.values('Z Grid K Unit Normal')
    for ipt in range(0,Zne.num_points):
        print("Point: {}".format(ipt))
        BL=False
        D=0.00001#First distance; must be O(dataset cell)
        Dm1=D
        while BL == False:
            Dm2=Dm1
            Dm1=D
            xp=x[ipt]+xn[ipt]*D
            yp=y[ipt]+yn[ipt]*D
            zp=z[ipt]+zn[ipt]*D
            (inBL,outBL,BL,out)=BL95(xp,yp,zp,v0,iVmag)
            if inBL == False:
                if outBL == False:
                    if BL == False:
                        D=0
            if inBL == True:
                D=D*2
            if outBL == True:
                D=D/2
            if out==True:
                print("probe out of the domain, process next point")
                break
            if Dm2 == D:
                D=(Dm2+Dm1)/2
                #print ("Back N' forth\n new D: {}".format(D))
            #print ("Pt:{}; D:{}".format(ipt,D))
            #print ("inBL:{}; outBL:{} ;BL:{}".format(inBL,outBL,BL))
            #input()
        Zne.values('BL_Thickness')[ipt]=D
        resultZne.values('x')[ipt]=xp
        resultZne.values('y')[ipt]=yp
        resultZne.values('z')[ipt]=zp

        
      
#saves the result
variablesToSave = [ds.variable(V) for V in ('x','y','z','BL_Thickness')]
zonesToSave = [Z for Z in ds.zones()]
tp.data.save_tecplot_plt(outfile, dataset=ds,
                                variables=variablesToSave,
                                zones=zonesToSave)
