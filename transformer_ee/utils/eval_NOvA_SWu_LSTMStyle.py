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
# model_name = "SWu_fd_fhc_LSTMStyle"
# filepath = "/exp/dune/app/users/jbarrow/nova_transformer_swu/fd_fhc/model_fd_fhc/result.npz"
# plotpath = "/exp/dune/app/users/jbarrow/nova_transformer_swu/fd_fhc/model_fd_fhc/plots/"

# model_name = "SWu_fd_rhc_LSTMStyle"
# filepath = "/exp/dune/app/users/jbarrow/nova_transformer_swu/fd_rhc/model_fd_rhc/result.npz"
# plotpath = "/exp/dune/app/users/jbarrow/nova_transformer_swu/fd_rhc/model_fd_rhc/plots/"

# model_name = "SWu_nd_fhc_LSTMStyle"
# filepath = "/exp/dune/app/users/jbarrow/nova_transformer_swu/nd_fhc/model_nd_fhc/result.npz"
# plotpath = "/exp/dune/app/users/jbarrow/nova_transformer_swu/nd_fhc/model_nd_fhc/plots/"

model_name = "SWu_nd_rhc_LSTMStyle"
filepath = "/exp/dune/app/users/jbarrow/nova_transformer_swu/nd_rhc/model_nd_rhc/result.npz"
plotpath = "/exp/dune/app/users/jbarrow/nova_transformer_swu/nd_rhc/model_nd_rhc/plots/"


print("Contents of the npz file:")
with np.load(filepath) as file:
  for key in file.keys():
      print(key)  

file = np.load(filepath)
trueval = file['trueval']
prediction = file['prediction']

# print("trueval shape: ",trueval.shape,prediction.shape)
# n_val,dim = trueval.shape

print("Original trueval shape: ", trueval.shape)
print("Original prediction shape: ", prediction.shape)

# # Apply the condition E_nu_true < 1.0
# mask = trueval[:, 0] < 1.0
# trueval = trueval[mask]
# prediction = prediction[mask]

print("Filtered trueval shape: ", trueval.shape)
print("Filtered prediction shape: ", prediction.shape)
n_val, dim = trueval.shape

#True Variables
E_nu_true = trueval[:,0]
E_l_true = trueval[:,1]

#Predicted variables
E_nu_pred = prediction[:,0]
E_l_pred = prediction[:,1]

#Make resolution variables
#Energy of the neutrino
E_nu_res_percent=100.*((E_nu_pred-E_nu_true)/E_nu_true)
E_nu_res_percent_diff=100.*((E_nu_pred-E_nu_true)/((E_nu_true+E_nu_pred)/2))
E_nu_res_abs_diff=E_nu_pred-E_nu_true
#Energy of the lepton
E_l_res_percent=100.*((E_l_pred-E_l_true)/E_l_true)
E_l_res_percent_diff=100.*((E_l_pred-E_l_true)/((E_l_true+E_l_pred)/2))
E_l_res_abs_diff=E_l_pred-E_l_true


# Start plotting

#1D resolutions as a function of true variables
#Energy of the neutrino
binstat.plot_xstat(E_nu_true,E_nu_res_percent,name=plotpath +"/E_nu_res_1D_percent",title="Neutrino Energy Reconstruction",scale='linear',xlabel="True Neutrino Energy",ylabel="Resolution on Neutrino Energy, Percent",range=(0.0,7.1))
plt.close()
binstat.plot_xstat(E_nu_true,E_nu_res_percent_diff,name=plotpath +"/E_nu_res_1D_percent_diff",title="Neutrino Energy Reconstruction",scale='linear',xlabel="True Neutrino Energy",ylabel="Resolution on Neutrino Energy, Percent Difference",range=(0.0,7.1))
plt.close()
binstat.plot_xstat(E_nu_true,E_nu_res_abs_diff,name=plotpath +"/E_nu_res_1D_abs_diff",title="Neutrino Energy Reconstruction",scale='linear',xlabel="True Neutrino Energy",ylabel="Resolution on Neutrino Energy, Absolute Difference",range=(0.0,7.1))
plt.close()

#Energy of the lepton
binstat.plot_xstat(E_l_true,E_l_res_percent,name=plotpath +"/E_l_res_1D_percent",xrange=(-1.0,1.0),yrange=(-1.0,1.0),title="Lepton Energy",xlabel="Lepton Energy",ylabel="Resolution on Lepton Energy, Percent")
plt.close()
binstat.plot_xstat(E_l_true,E_l_res_percent_diff,name=plotpath +"/E_l_res_1D_percent_diff",xrange=(-1.0,1.0),yrange=(-1.0,1.0),title="Lepton Energy",xlabel="Lepton Energy",ylabel="Resolution on Lepton Energy, Percent Difference")
plt.close()
binstat.plot_xstat(E_l_true,E_l_res_abs_diff,name=plotpath +"/E_l_res_1D_abs_diff",xrange=(-1.0,1.0),yrange=(-1.0,1.0),title="Lepton Energy",xlabel="Lepton Energy",ylabel="Resolution on Lepton Energy, Absolute Difference")
plt.close()

#1D resolutions
#Energy of the neutrino
binstat.plot_y_hist(E_nu_res_percent,name=plotpath +"/E_nu_res_percent",range=(-100.0,100.0),bins=500)
plt.close()
binstat.plot_y_hist(E_nu_res_percent_diff,name=plotpath +"/E_nu_res_percent_diff",range=(-100.0,100.0),bins=500)
plt.close()
binstat.plot_y_hist(E_nu_res_abs_diff,name=plotpath+"/E_nu_res_abs_diff",range=(-2.0,2.0),bins=500)
plt.close()

#Energy of the lepton
binstat.plot_y_hist(E_l_res_percent,name=plotpath +"/E_l_res_percent",range=(-100.0,100.0),bins=500,log=True)
plt.close()
binstat.plot_y_hist(E_l_res_percent_diff,name=plotpath +"/E_l_res_percent_diff",range=(-100.0,100.0),bins=500,log=True)
plt.close()
binstat.plot_y_hist(E_l_res_abs_diff,name=plotpath+"/E_l_res_abs_diff",range=(-2.0,2.0),bins=500,log=True)
plt.close()

#1D true and predicted variable distributions for comparisons
#Energy
binstat.plot_y_hist(E_nu_true,E_nu_pred,name=plotpath +"/E_nu_true_vs_pred_1D",range=(0.,7.1),bins=500,log=True,labels=["True Neutrino Energy","Pred. Neutrino Energy"])
plt.close()

#Energy of the lepton
binstat.plot_y_hist(E_l_true,E_l_pred,name=plotpath +"/E_l_true_vs_pred_1D",range=(0.,7.1),bins=500,log=True,labels=["True Lepton Energy","Pred. Lepton Energy"])
plt.close()


#2D plots
#True vs. predicted variables
#Energy
binstat.plot_2d_hist_count(E_nu_true,E_nu_pred,name=plotpath +"/E_nu_2D_true_vs_pred",xrange=(0.0,7.1),yrange=(0.0,7.1),xbins=400,ybins=400,title="Neutrino Energy Reconstruction",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Predicted Neutrino Energy (GeV)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_true,E_nu_pred,name=plotpath +"/E_nu_2D_true_vs_pred_wContours",xrange=(0.0,7.1),yrange=(0.0,7.1),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Neutrino Energy Reconstruction",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Predicted Neutrino Energy (GeV)",contours=True,contour_labels=True,histogram=True,diag_line=True)
plt.close()

#Energy of the lepton
binstat.plot_2d_hist_count(E_l_true,E_l_pred,name=plotpath +"/E_l_2D_true_vs_pred",xrange=(0.,8.0),yrange=(0.,8.0),xbins=400,ybins=400,title="Lepton Energy Reconstruction",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Predicted Lepton Energy (GeV)")
plt.close()
binstat.plot_2d_hist_contour(E_l_true,E_l_pred,name=plotpath +"/E_l_2D_true_vs_pred_wContours",xrange=(0.,8.0),yrange=(0.,8.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Lepton Energy Reconstruction",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Predicted Lepton Energy (GeV)",contours=True,contour_labels=True,histogram=True,diag_line=True)
plt.close()

#Neutrino Energy resolutions as a function of lepton resolutions, all different types
binstat.plot_2d_hist_count(E_nu_res_percent,E_l_res_percent,name=plotpath +"/E_nu_res_percent_vs_E_l_res_percent",xrange=(-100.0,100.0),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent (%)",ylabel="Lepton Energy Resolution, Percent (%)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_res_percent,E_l_res_percent,name=plotpath +"/E_nu_res_percent_vs_E_l_res_percent_wContours",xrange=(-100.0,100.0),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent (%)",ylabel="Lepton Energy Resolution, Percent (%)",contours=True,contour_labels=True,histogram=True,circle=True,radius=10)
plt.close()

binstat.plot_2d_hist_count(E_nu_res_percent,E_l_res_percent_diff,name=plotpath +"/E_nu_res_percent_vs_E_l_res_percent_diff",xrange=(-100.0,100.0),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent (%)",ylabel="Lepton Energy Resolution, Percent Difference (%)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_res_percent,E_l_res_percent_diff,name=plotpath +"/E_nu_res_percent_vs_E_l_res_percent_diff_wContours",xrange=(-100.0,100.0),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent (%)",ylabel="Lepton Energy Resolution, Percent Difference (%)",contours=True,contour_labels=True,histogram=True,circle=True,radius=10)
plt.close()

binstat.plot_2d_hist_count(E_nu_res_percent,E_l_res_abs_diff,name=plotpath +"/E_nu_res_percent_vs_E_l_res_abs_diff",xrange=(-100.0,100.0),yrange=(-2.0,2.0),xbins=400,ybins=400,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent (%)",ylabel="Lepton Energy Resolution, Absolute Difference (Degrees)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_res_percent,E_l_res_abs_diff,name=plotpath +"/E_nu_res_percent_vs_E_l_res_abs_diff_wContours",xrange=(-100.0,100.0),yrange=(-2.0,2.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent (%)",ylabel="Lepton Energy Resolution, Absolute Difference (Degrees)",contours=True,contour_labels=True,histogram=True,circle=True,ellipse=True,x_semimajor=10,y_semiminor=0.250)
plt.close()

binstat.plot_2d_hist_count(E_nu_res_percent_diff,E_l_res_percent,name=plotpath +"/E_nu_res_percent_diff_vs_E_l_res_percent",xrange=(-100.0,100.0),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent Difference (%)",ylabel="Lepton Energy Resolution, Percent (%)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_res_percent_diff,E_l_res_percent,name=plotpath +"/E_nu_res_percent_diff_vs_E_l_res_percent_wContours",xrange=(-100.0,100.0),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent Difference (%)",ylabel="Lepton Energy Resolution, Percent (%)",contours=True,contour_labels=True,histogram=True,circle=True,radius=10)
plt.close()

binstat.plot_2d_hist_count(E_nu_res_percent_diff,E_l_res_percent_diff,name=plotpath +"/E_nu_res_percent_diff_vs_E_l_res_percent_diff",xrange=(-100.0,100.0),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent Difference (%)",ylabel="Lepton Energy Resolution, Percent Difference (%)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_res_percent_diff,E_l_res_percent_diff,name=plotpath +"/E_nu_res_percent_diff_vs_E_l_res_percent_diff_wContour",xrange=(-100.0,100.0),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent Difference (%)",ylabel="Lepton Energy Resolution, Percent Difference (%)",contours=True,contour_labels=True,histogram=True,circle=True,radius=10)
plt.close()

binstat.plot_2d_hist_count(E_nu_res_percent_diff,E_l_res_abs_diff,name=plotpath +"/E_nu_res_percent_diff_vs_E_l_res_abs_diff",xrange=(-100.0,100.0),yrange=(-2,2.0),xbins=400,ybins=400,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent Difference (%)",ylabel="Lepton Energy Resolution, Absolute Difference (Degrees)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_res_percent_diff,E_l_res_abs_diff,name=plotpath +"/E_nu_res_percent_diff_vs_E_l_res_abs_diff_wContour",xrange=(-100.0,100.0),yrange=(-2.0,2.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Percent Difference (%)",ylabel="Lepton Energy Resolution, Absolute Difference (Degrees)",contours=True,contour_labels=True,histogram=True,ellipse=True,x_semimajor=10,y_semiminor=0.250)
plt.close()

binstat.plot_2d_hist_count(E_nu_res_abs_diff,E_l_res_percent,name=plotpath +"/E_nu_res_abs_diff_vs_E_l_res_percent",xrange=(-2.,2.),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Absolute Difference (GeV)",ylabel="Lepton Energy Resolution, Percent (%)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_res_abs_diff,E_l_res_percent,name=plotpath +"/E_nu_res_abs_diff_vs_E_l_res_percent_wContour",xrange=(-2.,2.),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Absolute Difference (GeV)",ylabel="Lepton Energy Resolution, Percent (%)",contours=True,contour_labels=True,histogram=True,ellipse=True,x_semimajor=0.250,y_semiminor=10)
plt.close()

binstat.plot_2d_hist_count(E_nu_res_abs_diff,E_l_res_percent_diff,name=plotpath +"/E_nu_res_abs_diff_vs_E_l_res_percent_diff",xrange=(-2.,2.),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Absolute Difference (GeV)",ylabel="Lepton Energy Resolution, Percent Difference (%)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_res_abs_diff,E_l_res_percent_diff,name=plotpath +"/E_nu_res_abs_diff_vs_E_l_res_percent_diff_wContour",xrange=(-2.,2.),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Absolute Difference (GeV)",ylabel="Lepton Energy Resolution, Percent Difference (%)",contours=True,contour_labels=True,histogram=True,ellipse=True,x_semimajor=0.250,y_semiminor=10)
plt.close()

binstat.plot_2d_hist_count(E_nu_res_abs_diff,E_l_res_abs_diff,name=plotpath +"/E_nu_res_abs_diff_vs_E_l_res_abs_diff",xrange=(-2.,2.),yrange=(-2.0,2.0),xbins=400,ybins=400,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Absolute Difference (GeV)",ylabel="Lepton Energy Resolution, Absolute Difference (Degrees)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_res_abs_diff,E_l_res_abs_diff,name=plotpath +"/E_nu_res_abs_diff_vs_E_l_res_abs_diff_wContour",xrange=(-2.,2.),yrange=(-2.0,2.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Overall Performance of LSTM-Style Kinematic Reconstruction",scale='linear',zscale='log',xlabel="Neutrino Energy Resolution, Absolute Difference (GeV)",ylabel="Lepton Energy Resolution, Absolute Difference (Degrees)",contours=True,contour_labels=True,histogram=True,ellipse=True,x_semimajor=0.250,y_semiminor=0.250)
plt.close()




#Energy resolution vs true energy
binstat.plot_2d_hist_count(E_nu_true,E_nu_res_percent,name=plotpath +"/E_nu_true_vs_E_nu_res_percent",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Neutrino Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Energy Resolution, Percent (%)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_true,E_nu_res_percent,name=plotpath +"/E_nu_true_vs_E_nu_res_percent_wContour",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Neutrino Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Energy Resolution, Percent (%)",contours=True,contour_labels=True,histogram=True)
plt.close()

binstat.plot_2d_hist_count(E_nu_true,E_nu_res_percent_diff,name=plotpath +"/E_nu_true_vs_E_nu_res_percent_diff",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Neutrino Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Energy Resolution, Percent Difference (%)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_true,E_nu_res_percent_diff,name=plotpath +"/E_nu_true_vs_E_nu_res_percent_diff_wContour",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Neutrino Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Energy Resolution, Percent Difference (%)",contours=True,contour_labels=True,histogram=True)
plt.close()

binstat.plot_2d_hist_count(E_nu_true,E_nu_res_abs_diff,name=plotpath +"/E_nu_true_vs_E_nu_res_abs_diff",xrange=(0.0,7.1),yrange=(-2.0,2.0),xbins=400,ybins=400,title="Neutrino Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Energy Resolution, Absolute Difference")
plt.close()
binstat.plot_2d_hist_contour(E_nu_true,E_nu_res_abs_diff,name=plotpath +"/E_nu_true_vs_E_nu_res_abs_diff_wContour",xrange=(0.0,7.1),yrange=(-2.0,2.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Neutrino Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Energy Resolution, Absolute Difference",contours=True,contour_labels=True,histogram=True)
plt.close()






#Lepton energy resolution vs true neutrino energy
binstat.plot_2d_hist_count(E_nu_true,E_l_res_percent,name=plotpath +"/E_nu_true_vs_E_l_res_percent",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Lepton Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Lepton Energy Resolution, Percent (%)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_true,E_l_res_percent,name=plotpath +"/E_nu_true_vs_E_l_res_percent_wContour",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Lepton Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Lepton Energy Resolution, Percent (%)",contours=True,contour_labels=True,histogram=True)
plt.close()

binstat.plot_2d_hist_count(E_nu_true,E_l_res_percent_diff,name=plotpath +"/E_nu_true_vs_E_l_res_percent_diff",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Lepton Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Lepton Energy Resolution, Percent Difference (%)")
plt.close()
binstat.plot_2d_hist_contour(E_nu_true,E_l_res_percent_diff,name=plotpath +"/E_nu_true_vs_E_l_res_percent_diff_wContour",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Lepton Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Lepton Energy Resolution, Percent Difference (%)",contours=True,contour_labels=True,histogram=True)
plt.close()

binstat.plot_2d_hist_count(E_nu_true,E_l_res_abs_diff,name=plotpath +"/E_nu_true_vs_E_l_res_abs_diff",xrange=(0.0,7.1),yrange=(-2.0,2.0),xbins=400,ybins=400,title="Lepton Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Lepton Energy Resolution, Absolute Difference")
plt.close()
binstat.plot_2d_hist_contour(E_nu_true,E_l_res_abs_diff,name=plotpath +"/E_nu_true_vs_E_l_res_abs_diff_wContour",xrange=(0.0,1.0),yrange=(-2.0,2.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Lepton Energy Reconstruction Performance Across Neutrino Energies",scale='linear',zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Lepton Energy Resolution, Absolute Difference",contours=True,contour_labels=True,histogram=True)
plt.close()







#Lepton energy resolution vs true lepton energy
binstat.plot_2d_hist_count(E_l_true,E_l_res_percent,name=plotpath +"/E_l_true_vs_E_l_res_percent",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=200,title="Lepton Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Lepton Energy Resolution, Percent (%)")
plt.close()
binstat.plot_2d_hist_contour(E_l_true,E_l_res_percent,name=plotpath +"/E_l_true_vs_E_l_res_percent_wContour",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=200,xbins_contour=35,ybins_contour=25,title="Lepton Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Lepton Energy Resolution, Percent (%)",contours=True,contour_labels=True,histogram=True)
plt.close()

binstat.plot_2d_hist_count(E_l_true,E_l_res_percent_diff,name=plotpath +"/E_l_true_vs_E_l_res_percent_diff",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=200,title="Lepton Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Lepton Energy Resolution, Percent Difference (%)")
plt.close()
binstat.plot_2d_hist_contour(E_l_true,E_l_res_percent_diff,name=plotpath +"/E_l_true_vs_E_l_res_percent_diff_wContour",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=200,xbins_contour=35,ybins_contour=25,title="Lepton Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Lepton Energy Resolution, Percent Difference (%)",contours=True,contour_labels=True,histogram=True)
plt.close()

binstat.plot_2d_hist_count(E_l_true,E_l_res_abs_diff,name=plotpath +"/E_l_true_vs_E_l_res_abs_diff",xrange=(0.0,7.1),yrange=(-2.0,2.0),xbins=400,ybins=200,title="Lepton Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Lepton Energy Resolution, Absolute Difference")
plt.close()
binstat.plot_2d_hist_contour(E_l_true,E_l_res_abs_diff,name=plotpath +"/E_l_true_vs_E_l_res_abs_diff_wContour",xrange=(0.0,7.1),yrange=(-2.0,2.0),xbins=400,ybins=200,xbins_contour=35,ybins_contour=25,title="Lepton Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Lepton Energy Resolution, Absolute Difference",contours=True,contour_labels=True,histogram=True)
plt.close()






#Energy resolution vs true theta
binstat.plot_2d_hist_count(E_l_true,E_nu_res_percent,name=plotpath +"/E_l_true_vs_E_nu_res_percent",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Neutrino Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Energy Resolution, Percent (%)")
plt.close()
binstat.plot_2d_hist_contour(E_l_true,E_nu_res_percent,name=plotpath +"/E_l_true_vs_E_nu_res_percent_wContour",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Neutrino Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Energy Resolution, Percent (%)",contours=True,contour_labels=True,histogram=True)
plt.close()

binstat.plot_2d_hist_count(E_l_true,E_nu_res_percent_diff,name=plotpath +"/E_l_true_vs_E_nu_res_percent_diff",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,title="Neutrino Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Energy Resolution, Percent Difference (%)")
plt.close()
binstat.plot_2d_hist_contour(E_l_true,E_nu_res_percent_diff,name=plotpath +"/E_l_true_vs_E_nu_res_percent_diff_wContour",xrange=(0.0,7.1),yrange=(-100.0,100.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Neutrino Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Energy Resolution, Percent Difference (%)",contours=True,contour_labels=True,histogram=True)
plt.close()

binstat.plot_2d_hist_count(E_l_true,E_nu_res_abs_diff,name=plotpath +"/E_l_true_vs_E_nu_res_abs_diff",xrange=(0.0,7.1),yrange=(-2.0,2.0),xbins=400,ybins=400,title="Neutrino Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Energy Resolution, Absolute Difference")
plt.close()
binstat.plot_2d_hist_contour(E_l_true,E_nu_res_abs_diff,name=plotpath +"/E_l_true_vs_E_nu_res_abs_diff_wContour",xrange=(0.0,7.1),yrange=(-2.0,2.0),xbins=400,ybins=400,xbins_contour=35,ybins_contour=35,title="Neutrino Energy Reconstruction Performance Across Lepton Energies",scale='linear',zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Energy Resolution, Absolute Difference",contours=True,contour_labels=True,histogram=True)
plt.close()



