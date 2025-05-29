# **README: From producing CSV files & Running training to Plotting**  

## **Step 1: CSV and Signal Selection Files**  

### **CSV File Organization**  
The work is divided based on whether leptons are treated as **scalars** or **vectors**, as described below:  

- **Vector Leptons without NC** – Only charged leptons enter as vectors; neutral leptons are excluded.  
- **Vector Leptons with NC** – Both charged and neutral leptons enter as vectors.  
- **Vector Leptons with 0 NC** – Charged leptons enter as vectors, while neutral leptons have all kinematic values set to zero.  
- **Scalar Leptons with 0 NC** – True kinematic information for charged leptons entering as scalars; neutral leptons have kinematic values set to zero.  
- **Scalar Leptons with NC** – Both charged and neutral leptons enter as scalars.  

---

### **Signal Selection Format**  

**Sample Signal Selection File:**  

```
EventNum: All            # Use 'All' to process all events or specify a number for a subset  
Initial_Nu_Energy_min: 0.1  
Initial_Nu_Energy_max: 1  
Initial_PDG: Any         # Use 'Any' for all neutrinos or specify a PDG code for a specific neutrino  
Final_state_lepton_PDG_code_or_process: Inclusive  # Use 'CC' for charged leptons, 'NC' for neutral leptons, or 'Inclusive' for both.  
Proton_KE: 0.025         # Kinetic energy in GeV  
Pi+-_KE: 0.07  
K+-_KE: 0.03  
Muon_KE: 0.005  
Electron_KE: 0.005  
num_proton: N            # Use 'N' for any number of protons/pions or specify a value  
num_pion: N  
INPUT_ROOT_FILE: NNBarAtm_hA_BR.100000000.gtrac.root  
OUTPUT_ROOT_FILE: AnyNu_Inclusive_Thresh_p1to1_  
OUTPUT_DIR: AnyNu_Inclusive_Thresh_p1to1_  
OUTPUT_NAME: AnyNu_Inclusive_Thresh_p1to1_  
```  

---

### **Running C++ Files**  

To run a single signal selection file with the C++ code, use the following command:  
```bash
root -b -q 'ScalarLept_wNC.C("Signal_Selection1.txt")'
```  
**Note:**  
- Use the provided bash script to process multiple text files simultaneously.  

---