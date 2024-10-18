import os
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.optimize import curve_fit
import binstat

#Make file paths
model_name1 = "SWu_fd_fhc_LSTMStyle"
file1path = "/exp/dune/app/users/jbarrow/nova_transformer_swu/fd_fhc/model_fd_fhc/result.npz"


model_name2 = "SWu_fd_rhc_LSTMStyle"
file2path = "/exp/dune/app/users/jbarrow/nova_transformer_swu/fd_rhc/model_fd_rhc/result.npz"


model_name3 = "SWu_nd_fhc_LSTMStyle"
file3path = "/exp/dune/app/users/jbarrow/nova_transformer_swu/nd_fhc/model_nd_fhc/result.npz"

model_name4 = "SWu_nd_rhc_LSTMStyle"
file4path = "/exp/dune/app/users/jbarrow/nova_transformer_swu/nd_rhc/model_nd_rhc/result.npz"

plotpath = "/exp/dune/app/users/jbarrow/nova_transformer_swu/comparison_plots/"


#Load files and dataframes
print("Contents of the npz file1:")
with np.load(file1path) as file1:
  for key in file1.keys():
      print(key)

print("Contents of the npz file2:")
with np.load(file2path) as file2:
  for key in file2.keys():
      print(key)

print("Contents of the npz file3:")
with np.load(file3path) as file3:
  for key in file3.keys():
      print(key)

print("Contents of the npz file4:")
with np.load(file4path) as file4:
  for key in file4.keys():
      print(key) 

file1 = np.load(file1path)
trueval_fd_fhc = file1['trueval']
prediction_fd_fhc = file1['prediction']

file2 = np.load(file2path)
trueval_fd_rhc = file2['trueval']
prediction_fd_rhc = file2['prediction']

file3 = np.load(file3path)
trueval_nd_fhc = file3['trueval']
prediction_nd_fhc = file3['prediction']

file4 = np.load(file4path)
trueval_nd_rhc = file4['trueval']
prediction_nd_rhc = file4['prediction']


#Print dataframe shapes and sizes
print("trueval_fd_fch shape: ",trueval_fd_fhc.shape,prediction_fd_fhc.shape)
n_val1,dim1 = trueval_fd_fhc.shape
print("trueval_fd_rch shape: ",trueval_fd_rhc.shape,prediction_fd_rhc.shape)
n_val2,dim2 = trueval_fd_rhc.shape
print("trueval_nd_fch shape: ",trueval_nd_fhc.shape,prediction_nd_fhc.shape)
n_val3,dim3 = trueval_nd_fhc.shape
print("trueval_nd_rch shape: ",trueval_nd_rhc.shape,prediction_nd_rhc.shape)
n_val4,dim4 = trueval_nd_rhc.shape

#True neutrino variables
E_nu_true_fd_fhc = trueval_fd_fhc[:,0]
E_nu_true_fd_rhc = trueval_fd_rhc[:,0]
E_nu_true_nd_fhc = trueval_nd_fhc[:,0]
E_nu_true_nd_rhc = trueval_nd_rhc[:,0]
#True lepton variables
E_l_true_fd_fhc = trueval_fd_fhc[:,1]
E_l_true_fd_rhc = trueval_fd_rhc[:,1]
E_l_true_nd_fhc = trueval_nd_fhc[:,1]
E_l_true_nd_rhc = trueval_nd_rhc[:,1]
#Predicted neutrino variables
E_nu_pred_fd_fhc = prediction_fd_fhc[:,0]
E_nu_pred_fd_rhc = prediction_fd_rhc[:,0]
E_nu_pred_nd_fhc = prediction_nd_fhc[:,0]
E_nu_pred_nd_rhc = prediction_nd_rhc[:,0]
#Predicted lepton variables
E_l_pred_fd_fhc = prediction_fd_fhc[:,1]
E_l_pred_fd_rhc = prediction_fd_rhc[:,1]
E_l_pred_nd_fhc = prediction_nd_fhc[:,1]
E_l_pred_nd_rhc = prediction_nd_rhc[:,1]

#Make resolution variables
#Neutrino energy resolutions
E_nu_res_percent_diff_fd_fhc=100.*((E_nu_pred_fd_fhc-E_nu_true_fd_fhc)/((E_nu_true_fd_fhc+E_nu_pred_fd_fhc)/2))
E_nu_res_percent_diff_fd_rhc=100.*((E_nu_pred_fd_rhc-E_nu_true_fd_rhc)/((E_nu_true_fd_rhc+E_nu_pred_fd_rhc)/2))
E_nu_res_percent_diff_nd_fhc=100.*((E_nu_pred_nd_fhc-E_nu_true_nd_fhc)/((E_nu_true_nd_fhc+E_nu_pred_nd_fhc)/2))
E_nu_res_percent_diff_nd_rhc=100.*((E_nu_pred_nd_rhc-E_nu_true_nd_rhc)/((E_nu_true_nd_rhc+E_nu_pred_nd_rhc)/2))
#Lepton energy resolutions
E_l_res_percent_diff_fd_fhc=100.*((E_l_pred_fd_fhc-E_l_true_fd_fhc)/((E_l_true_fd_fhc+E_l_pred_fd_fhc)/2))
E_l_res_percent_diff_fd_rhc=100.*((E_l_pred_fd_rhc-E_l_true_fd_rhc)/((E_l_true_fd_rhc+E_l_pred_fd_rhc)/2))
E_l_res_percent_diff_nd_fhc=100.*((E_l_pred_nd_fhc-E_l_true_nd_fhc)/((E_l_true_nd_fhc+E_l_pred_nd_fhc)/2))
E_l_res_percent_diff_nd_rhc=100.*((E_l_pred_nd_rhc-E_l_true_nd_rhc)/((E_l_true_nd_rhc+E_l_pred_nd_rhc)/2))



binstat.plot_xstat(x_list=[E_nu_true_fd_fhc,E_nu_true_fd_rhc,E_nu_true_nd_fhc,E_nu_true_nd_rhc],y_list=[E_nu_res_percent_diff_fd_fhc,E_nu_res_percent_diff_fd_rhc,E_nu_res_percent_diff_nd_fhc,E_nu_res_percent_diff_nd_rhc],name=plotpath +"/E_nu_res_all",title="Neutrino Energy Reconstruction",xlabel="True Neutrino Energy",ylabel="Resolution on Neutrino Energy, Percent Diff. (%)",labels=["FD FHC","FD RHC","ND FHC","ND RHC"],colors=["red","pink","blue","indigo"],range=(0.,7.),legend_loc="bottomleft")
plt.close()

binstat.plot_xstat(x_list=[E_l_true_fd_fhc,E_l_true_fd_rhc,E_l_true_nd_fhc,E_l_true_nd_rhc],y_list=[E_l_res_percent_diff_fd_fhc,E_l_res_percent_diff_fd_rhc,E_l_res_percent_diff_nd_fhc,E_l_res_percent_diff_nd_rhc],name=plotpath +"/E_l_res_all",title="Lepton Energy Reconstruction",xlabel="True Final State Lepton Energy",ylabel="Resolution on Lepton Energy, Percent Diff. (%)",labels=["FD FHC","FD RHC","ND FHC","ND RHC"],colors=["red","pink","blue","indigo"],range=(0.,7.),legend_loc="bottomleft")
plt.close()

binstat.plot_xstat(x_list=[E_l_true_fd_fhc,E_l_true_fd_rhc,E_l_true_nd_fhc,E_l_true_nd_rhc],y_list=[E_l_res_percent_diff_fd_fhc,E_l_res_percent_diff_fd_rhc,E_l_res_percent_diff_nd_fhc,E_l_res_percent_diff_nd_rhc],name=plotpath +"/E_l_res_all_inner_range",title="Lepton Energy Reconstruction",xlabel="True Final State Lepton Energy",ylabel="Resolution on Lepton Energy, Percent Diff. (%)",labels=["FD FHC","FD RHC","ND FHC","ND RHC"],colors=["red","pink","blue","indigo"],range=(0.5,4.),legend_loc="bottomleft")
plt.close()

binstat.plot_y_hist(E_nu_res_percent_diff_fd_fhc,E_nu_res_percent_diff_fd_rhc,E_nu_res_percent_diff_nd_fhc,E_nu_res_percent_diff_nd_rhc,name=plotpath+"/E_nu_res_percent_diff_all",labels=["FD FHC", "FD RHC", "ND FHC", "ND RHC"],xrange=(-65.0,40.0),yrange=(1E-5,0.20),xlabel="Neutrino Energy Resolution, Percent Difference (%)",log=True,vline=True,bins=200,colors=["red","pink","blue","indigo"],normalize=True)
plt.close()

binstat.plot_y_hist(E_l_res_percent_diff_fd_fhc,E_l_res_percent_diff_fd_rhc,E_l_res_percent_diff_nd_fhc,E_l_res_percent_diff_nd_rhc,name=plotpath+"/E_l_res_percent_diff_all",labels=["FD FHC", "FD RHC", "ND FHC", "ND RHC"],xrange=(-65.0,40.0),yrange=(1E-5,0.20),xlabel="Lepton Energy Resolution, Percent Difference (%)",log=True,vline=True,bins=200,colors=["red","pink","blue","indigo"],normalize=True)
plt.close()

binstat.plot_2d_hist_percentile_contour(E_l_true_fd_fhc,E_l_pred_fd_fhc,name=plotpath+"/E_l_2D_true_vs_pred_fd_fhc_wPercentileContours_nohist",xrange=(0.,7.0),yrange=(0.,7.0),xbins=400,ybins=400,xbins_contour=100,ybins_contour=100,title="Lepton Energy Reconstruction",scale='linear', zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Predicted Lepton Energy (GeV)",contours=True,contour_labels=True,histogram=False,diag_line=True)
plt.close()

binstat.plot_2d_hist_percentile_contour(E_l_true_fd_fhc,E_l_pred_fd_fhc,name=plotpath+"/E_l_2D_true_vs_pred_fd_fhc_wPercentileContours_whist",xrange=(0.,7.0),yrange=(0.,7.0),xbins=400,ybins=400,xbins_contour=100,ybins_contour=100,title="Lepton Energy Reconstruction",scale='linear', zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Predicted Lepton Energy (GeV)",contours=True,contour_labels=True,histogram=True,diag_line=True)
plt.close()

binstat.plot_2d_hist_percentile_contour(E_nu_true_fd_fhc,E_nu_pred_fd_fhc,name=plotpath+"/E_nu_2D_true_vs_pred_fd_fhc_wPercentileContours_nohist",xrange=(0.,7.0),yrange=(0.,7.0),xbins=400,ybins=400,xbins_contour=100,ybins_contour=100,title="Neutrino Energy Reconstruction",scale='linear', zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Predicted Neutrino Energy (GeV)",contours=True,contour_labels=True,histogram=False,diag_line=True)
plt.close()

binstat.plot_2d_hist_percentile_contour(E_nu_true_fd_fhc,E_nu_pred_fd_fhc,name=plotpath+"/E_nu_2D_true_vs_pred_fd_fhc_wPercentileContours_whist",xrange=(0.,7.0),yrange=(0.,7.0),xbins=400,ybins=400,xbins_contour=100,ybins_contour=100,title="Neutrino Energy Reconstruction",scale='linear', zscale='log',xlabel="True Neutrino Energy (GeV)",ylabel="Predicted Neutrino Energy (GeV)",contours=True,contour_labels=True,histogram=True,diag_line=True)
plt.close()

# # Example usage for a single dataset
# x_single = np.random.normal(3, 1, 1000)
# y_single = np.random.normal(3, 1, 1000)
# plot_2d_hist_percentile_contour(
#     [x_single], [y_single], ["Dataset 1"],
#     name="single_contour",
#     xrange=(0., 7.0), yrange=(0., 7.0),
#     xbins_contour=100, ylabel="Y-axis", xlabel="X-axis",
#     contour_colors=['red'], line_widths=[3], contour_labels=True,
#     diag_line=True, circle=True, radius=3,
#     ellipse=True, x_semimajor=5, y_semiminor=3,
#     figsize=(8, 8), dpi=100
# )

# # Example usage for multiple datasets
# x_multi = [np.random.normal(3, 1, 1000), np.random.normal(4, 1, 1000)]
# y_multi = [np.random.normal(3, 1, 1000), np.random.normal(4, 1, 1000)]
# plot_2d_hist_percentile_contour(
#     x_multi, y_multi, ["Dataset 1", "Dataset 2"],
#     name="multiple_contours",
#     xrange=(0., 7.0), yrange=(0., 7.0),
#     xbins_contour=100, ylabel="Y-axis", xlabel="X-axis",
#     contour_colors=['red', 'blue'], line_widths=[3, 2], contour_labels=True,
#     diag_line=True, circle=True, radius=3,
#     ellipse=True, x_semimajor=5, y_semiminor=3,
#     figsize=(8, 8), dpi=100
# )


binstat.plot_2d_hist_percentile_multi_contour([E_l_true_fd_fhc,E_l_true_fd_rhc,E_l_true_nd_fhc,E_l_true_nd_rhc],[E_l_pred_fd_fhc,E_l_pred_fd_rhc,E_l_pred_nd_fhc,E_l_pred_nd_rhc],["FD FHC","FD RHC","ND FHC","ND RHC"],name=plotpath+"/E_l_2D_true_vs_pred_all_comp_wPercentileContours_nohist",xrange=(0.,7.0),yrange=(0.,7.0),xbins=400,ybins=400,xbins_contour=100,ybins_contour=100,title="Lepton Energy Reconstruction",scale='linear', zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Predicted Lepton Energy (GeV)",contours=True,contour_labels=False,diag_line=True,contour_colors=["red","pink","blue","indigo"],line_widths=[2,2,2,2],figsize=(8, 8), dpi=100)
plt.close()

binstat.plot_2d_hist_percentile_multi_contour([E_nu_true_fd_fhc,E_nu_true_fd_rhc,E_nu_true_nd_fhc,E_nu_true_nd_rhc],[E_nu_pred_fd_fhc,E_nu_pred_fd_rhc,E_nu_pred_nd_fhc,E_nu_pred_nd_rhc],["FD FHC","FD RHC","ND FHC","ND RHC"],name=plotpath+"/E_nu_2D_true_vs_pred_all_comp_wPercentileContours_nohist",xrange=(0.,7.0),yrange=(0.,7.0),xbins=400,ybins=400,xbins_contour=100,ybins_contour=100,title="Lepton Energy Reconstruction",scale='linear', zscale='log',xlabel="True Lepton Energy (GeV)",ylabel="Predicted Lepton Energy (GeV)",contour_labels=False,contour_colors=["red","pink","blue","indigo"],line_widths=[2,2,2,2],diag_line=True,figsize=(8, 8), dpi=100)
plt.close()