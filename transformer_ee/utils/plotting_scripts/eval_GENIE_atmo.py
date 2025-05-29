import os
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.optimize import curve_fit
import binstat



# Check if variable exists (optional)
# if 'model_path' in os.environ:  # Replace with the variable name you want to check
#     model_path = os.environ['model_path']
#     print("model_path : ",model_path)
# else:
#     print("model_path not found")

# filepath = model_path + "/result.npz"
model_name = "GENIEv3-0-6-Honda-Truth-hA-LFG_wLeptonScalars_MAE"
filepath = "/home/jbarrow/MLProject2/save/model/GENIEv3-0-6-Honda-Truth-hA-LFG_wLeptonScalars_MAE/model_GENIEv3-0-6-Honda-Truth-hA-LFG_wLeptonScalars_MAE/result.npz"
plotpath = "/home/jbarrow/MLProject2/save/model/GENIEv3-0-6-Honda-Truth-hA-LFG_wLeptonScalars_MAE/model_GENIEv3-0-6-Honda-Truth-hA-LFG_wLeptonScalars_MAE/"

print("Contents of the npz file:")
with np.load(filepath) as file:
  for key in file.keys():
      print(key)  
      
file = np.load(filepath)
trueval = file['trueval']
prediction = file['prediction']

print("trueval shape: ",trueval.shape,prediction.shape)
n_val,dim = trueval.shape

#True Variables
E_nu_true = trueval[:,0]
Px_nu_true = trueval[:,1]
Py_nu_true = trueval[:,2]
Pz_nu_true = trueval[:,3]
#True angle variables
Cos_Theta_nu_true = Py_nu_true / (((Px_nu_true) ** 2 + (Py_nu_true) ** 2 + (Pz_nu_true) **2) ** 0.5 )
#Theta_nu_true = math.acos(Cos_Theta_nu_true)

#Predicted variables
E_nu_pred = prediction[:,0]
Px_nu_pred = prediction[:,1]
Py_nu_pred = prediction[:,2]
Pz_nu_pred = prediction[:,3]
#Predicted angle variables
Cos_Theta_nu_pred = Py_nu_pred / (((Px_nu_pred) ** 2 + (Py_nu_pred) ** 2 + (Pz_nu_pred) **2) ** 0.5 )
#Theta_nu_pred = math.acos(Cos_Theta_nu_pred)

#binstat.plot_xstat(E_nu_true,E_nu_pred,name=plotpath +"/en",title=r'Transformer Performance for $E_{\nu}$ Reconstruction',scale='linear',xlabel=r'True $E_{\nu}$',ylabel=r'Predicted $E_{\nu}$')
binstat.plot_xstat(E_nu_true,E_nu_pred,name=plotpath +"/en",title="Atmospheric Neutrino Energy Reconstruction",scale='linear',xlabel="True Neutrino Energy",ylabel="Predicted Neutrino Energy")

#binstat.plot_xstat(Cos_Theta_nu_true,Cos_Theta_nu_pred,name=plotpath +"/ct",title="Transformer Performance for $cos(\theta_{\nu})$",xlabel="True $cos(\theta_{\nu})$",ylabel="Predicted $cos(\theta_{\nu})$")
binstat.plot_xstat(Cos_Theta_nu_true,Cos_Theta_nu_pred,name=plotpath +"/ct",title="Atmospheric Neutrinos' Cosine of Incoming Angle",xlabel="True Cosine of Incoming Angle",ylabel="Predicted Cosine of Incoming Angle")

binstat.plot_y_hist(E_nu_pred,name=plotpath+"/en_res")

#binstat.plot_2d_hist_count(E_nu_true,E_nu_pred,name=plotpath +"/en_hist2d",xrange=(0.0,4.0),yrange=(0.0,4.0),title="Transformer Performance for $E_{\nu}$ Reconstruction",scale='linear',xlabel="True $E_{\nu}$",ylabel="Predicted $E_{\nu}$")
binstat.plot_2d_hist_count(E_nu_true,E_nu_pred,name=plotpath +"/en_hist2d",xrange=(0.0,4.0),yrange=(0.0,4.0),title="Atmospheric Neutrino Energy Reconstruction",scale='linear',zscale='log',xlabel="True Neutrino Energy",ylabel="Predicted Neutrino Energy")

#binstat.plot_2d_hist_count(Px_nu_true,Px_nu_pred,name=model_name +"/ct_hist2d",xrange=(-1.,1.),yrange=(-1.,1.))

#binstat.plot_2d_hist_count(Cos_Theta_nu_true,Cos_Theta_nu_pred,name=plotpath +"/ct_hist2d",xrange=(-1.0,1.0),yrange=(-1.0,1.0),title="Transformer Performance for $cos(\theta_{\nu})$ Reconstruction",xlabel="True cos(\theta_{\nu})",ylabel="Predicted cos(\theta_{\nu})")
binstat.plot_2d_hist_count(Cos_Theta_nu_true,Cos_Theta_nu_pred,name=plotpath +"/ct_hist2d",xrange=(-1.0,1.0),yrange=(-1.0,1.0),title="Atmospheric Neutrinos' Cosine of Incoming Angle",scale='linear',zscale='log',xlabel="True Cosine of Incoming Angle",ylabel="Predicted Cosine of Incoming Angle")