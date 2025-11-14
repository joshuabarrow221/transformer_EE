#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <variant>
#include <vector>
#include <TFile.h>
#include <TH1.h>
#include <TTree.h>
#include "TCanvas.h"
#include "TStyle.h"
#include <tuple>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
// Function to convert a string to the appropriate type
// Function to convert a string to the appropriate type
const double PI = 3.141592653589793;
const double R = 6371000;
const double h = 15000;

TH1D *Init_Nu_Mom;
TH1D *Init_Nu_Theta;
TH1D *Init_Nu_CosTheta;
TH1D *Init_Nu_Phi;
TH2D *Oscillogram;
TH1D *Fin_CC_Lept_Mom;
TH1D *Fin_CC_Lept_Theta;
TH1D *Fin_CC_Lept_CosTheta;
TH1D *Fin_NC_Lept_Mom;
TH1D *Fin_NC_Lept_Theta;
TH1D *Fin_NC_Lept_CosTheta;
TH1D *Fin_Prot_Mom;
TH1D *Fin_PiPlus_Mom;
TH1D *Fin_PiMinus_Mom;
TH1D *Fin_PiZero_Mom;
TH1D *Fin_Prot_Mult;
TH1D *Fin_PiPlus_Mult;
TH1D *Fin_PiMinus_Mult;
TH1D *Fin_PiZero_Mult;
TH1D *Fin_Gamma_Mom;

std::vector<std::string> split(const std::string &str, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

// using DictionaryType = std::map<std::string, std::variant<int, float, std::string>>;
// std::variant<int, float, std::string> getValue(const DictionaryType& dictionary, const std::string& key) {
//     const auto& value = dictionary.at(key);

//     if (std::holds_alternative<int>(value)) {
//         return std::get<int>(value);
//     } else if (std::holds_alternative<float>(value)) {
//         return std::get<float>(value);
//     } else if (std::holds_alternative<std::string>(value)) {
//         return std::get<std::string>(value);
//     } else {
//         throw std::runtime_error("Unexpected type encountered for key '" + key + "'.");
//     }
// }

std::variant<int, float, std::string> convert_value(const std::string &value)
{
    // Try to convert to integer
    std::istringstream intStream(value);
    int intValue;
    if (intStream >> intValue && intStream.eof())
    {
        // //std::cout << "conversion, returning int: " << value << std::endl;
        return intValue;
    }

    // If conversion to integer fails, try to convert to float
    std::istringstream floatStream(value);
    float floatValue;
    if (floatStream >> floatValue && floatStream.eof())
    {
        // //std::cout << "conversion, returning float: " << value << std::endl;
        return floatValue;
    }

    // If conversion to both integer and float fails, return the original string
    // //std::cout << "No conversion, returning string: " << value << std::endl;
    return value;
}

// Function to process the input file and return a dictionary
std::map<std::string, std::variant<int, float, std::string>> process_file(const std::string &input_file)
{
    std::ifstream infile(input_file);
    std::map<std::string, std::variant<int, float, std::string>> dictionary;

    std::string line;
    while (std::getline(infile, line))
    {
        size_t delimiter_pos = line.find(':');
        if (delimiter_pos != std::string::npos)
        {
            std::string key = line.substr(0, delimiter_pos);
            std::string value = line.substr(delimiter_pos + 1);

            // Trim leading and trailing spaces from key and value
            size_t start = key.find_first_not_of(" \t\r\n");
            size_t end = key.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos)
            {
                key = key.substr(start, end - start + 1);
            }
            start = value.find_first_not_of(" \t\r\n");
            end = value.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos)
            {
                value = value.substr(start, end - start + 1);
            }

            // Convert value and store in dictionary
            dictionary[key] = convert_value(value);
        }
    }

    infile.close();
    return dictionary;
}

std::string variantToString(const std::variant<int, float, std::string> &value)
{
    if (std::holds_alternative<int>(value))
    {
        return std::to_string(std::get<int>(value));
    }
    else if (std::holds_alternative<float>(value))
    {
        return std::to_string(std::get<float>(value));
    }
    else if (std::holds_alternative<std::string>(value))
    {
        return std::get<std::string>(value);
    }
    throw std::runtime_error("Unsupported type for conversion to string");
}

std::tuple<double, double, double> kinematics(const double tStdHepP4[], int j)
{
    double fAbsoluteParticleMomentum = sqrt(pow(tStdHepP4[4 * j], 2) + pow(tStdHepP4[4 * j + 1], 2) + pow(tStdHepP4[4 * j + 2], 2));
    double fInvMass = sqrt(pow(tStdHepP4[4 * j + 3], 2) - pow(fAbsoluteParticleMomentum, 2));
    double fKE = tStdHepP4[4 * j + 3] - fInvMass;

    return std::make_tuple(fAbsoluteParticleMomentum, fInvMass, fKE);
}

// function to print the particle level info to outfile
void writeVectorToFile(std::ofstream &outfile, const std::vector<double> &values)
{
    for (int j = 0; j < values.size(); j++)
    {
        // Be sure to end the vector with a quote
        if (j < values.size() - 1)
        {
            outfile << std::setprecision(6) << values[j] << ",";
        }
        if (j == values.size() - 1)
        {
            outfile << std::setprecision(6) << values[j] << "\"";
        }
    }
    outfile << ",\"";
}
void getValue(const std::map<std::string, std::variant<int, float, std::string>> &dictionary,
              const std::string &key,
              std::variant<int, float, std::string> &name)
{

    const auto &value = dictionary.at(key);

    // Assigning value to name based on its type
    if (std::holds_alternative<int>(value))
    {
        name = std::get<int>(value);
    }
    else if (std::holds_alternative<float>(value))
    {
        name = std::get<float>(value);
    }
    else if (std::holds_alternative<std::string>(value))
    {
        name = std::get<std::string>(value);
    }
}

double getDoubleValue(const std::variant<int, float, std::string> &value)
{
    if (std::holds_alternative<int>(value))
    {
        return static_cast<double>(std::get<int>(value));
    }
    else if (std::holds_alternative<float>(value))
    {
        return static_cast<double>(std::get<float>(value));
    }
    throw std::runtime_error("Unsupported type for conversion to double");
}
std::tuple<double, double> kinematics_massless(const double tStdHepP4[], int j)
{
    // double fAbsoluteParticleMomentum = sqrt(pow(tStdHepP4[4 * j], 2) + pow(tStdHepP4[4 * j + 1], 2) + pow(tStdHepP4[4 * j + 2], 2));
    double fAbsoluteParticleMomentum = tStdHepP4[4 * j + 3];
    // std::cout << "Massless energy: " << tStdHepP4[4 * j + 3] << std::endl;
    double fKE = fAbsoluteParticleMomentum;

    return std::make_tuple(fAbsoluteParticleMomentum, fKE);
}

double calc_baseline(const double tStdHepP4[], double fAbsoluteParticleMomentum, int j)
{
    double theta_z = acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
    double eta = PI - theta_z;
    double baseline = (sqrt(pow((R + h), 2) - pow((R * sin(eta)), 2)) + (R * cos(eta))) / 1000.0;

    return baseline;
}

double phi_nu(const double tStdHepP4[], double fAbsoluteParticleMomentum, int j)
{
    double num = sqrt(pow(tStdHepP4[4 * j], 2) + pow(tStdHepP4[4 * j + 2], 2));
    double phi = acos(num / fAbsoluteParticleMomentum);

    return (180. / PI) * phi;
}

// void lepton_kinematics(const double tStdHepP4[],int j,const int tStdHepPdg[],
//                     const std::map<std::string, std::variant<int,float, std::string>>& dictionary){

//     auto [fAbsoluteParticleMomentum, fInvMass,fKE] = kinematics(tStdHepP4, j);

//     double muon_ke=getDoubleValue(dictionary.at("Muon_KE"));
//     double electron_ke=getDoubleValue(dictionary.at("Electron_KE"));

//     //std::cout<<"Lepton_mom is "<<fAbsoluteParticleMomentum<<" and fKE_lept is  "<<fKE<< endl;

//     if ((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE >= muon_ke ||
//         (tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE >= electron_ke) {
//         // //std::cout << "lepton loop for j: " << j << " and status: " << tStdHepStatus[j] <<" and pdg " << tStdHepPdg[j]<< std::endl;

//         pdgs.push_back(tStdHepPdg[j]);
//         masses.push_back(fInvMass);
//         energies.push_back(fKE);
//         pxs.push_back(tStdHepP4[4 * j]);
//         pys.push_back(tStdHepP4[4 * j + 1]);
//         pzs.push_back(tStdHepP4[4 * j + 2]);
//         costheta_arr.push_back(tStdHepP4[4*j+1]/fAbsoluteParticleMomentum);
//         theta_arr.push_back((180./PI)*acos(tStdHepP4[4*j+1]/fAbsoluteParticleMomentum));

//         tot_fKE += fKE;
//         tot_fpx += tStdHepP4[4 * j];
//         tot_fpy += tStdHepP4[4 * j + 1];
//         tot_fpz += tStdHepP4[4 * j + 2];

//         Fin_CC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
//         Fin_CC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
//         Fin_CC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
//     }

//     else {

//         pdgs.push_back(tStdHepPdg[j]);
//         masses.push_back(fInvMass);
//         energies.push_back(fKE);
//         pxs.push_back(tStdHepP4[4 * j]);
//         pys.push_back(tStdHepP4[4 * j + 1]);
//         pzs.push_back(tStdHepP4[4 * j + 2]);
//         costheta_arr.push_back(tStdHepP4[4*j+1]/fAbsoluteParticleMomentum);
//         theta_arr.push_back((180./PI)*acos(tStdHepP4[4*j+1]/fAbsoluteParticleMomentum));

//         tot_fKE += fKE;
//         tot_fpx += tStdHepP4[4 * j];
//         tot_fpy += tStdHepP4[4 * j + 1];
//         tot_fpz += tStdHepP4[4 * j + 2];

//         Fin_CC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
//         Fin_CC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
//         Fin_CC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));

//     }
// }

void finalparticles_info(const double tStdHepP4[], int j, const int tStdHepPdg[], std::vector<int> &pdgs, std::vector<double> &masses,
                         std::vector<double> &energies, std::vector<double> &pxs,
                         std::vector<double> &pys, std::vector<double> &pzs, std::vector<double> &costheta_arr, std::vector<double> &theta_arr, double &tot_fKE, double &tot_hadronic_energy, double &tot_fpx, double &tot_fpy, double &tot_fpz,
                         const std::map<std::string, std::variant<int, float, std::string>> &dictionary)
{
    double muon_ke = getDoubleValue(dictionary.at("Muon_KE"));
    double electron_ke = getDoubleValue(dictionary.at("Electron_KE"));
    double prot_ke = getDoubleValue(dictionary.at("Proton_KE"));
    double kaon_ke = getDoubleValue(dictionary.at("K+-_KE"));
    double pion_ke = getDoubleValue(dictionary.at("Pi+-_KE"));
    auto [fAbsoluteParticleMomentum, fInvMass, fKE] = kinematics(tStdHepP4, j);
    auto [fAbsoluteParticleMomentum_gamma, fKE_gamma] = kinematics_massless(tStdHepP4, j);
    // define costheta and theta array

    if ((tStdHepPdg[j] == 2212 && fKE >= prot_ke) ||
        (tStdHepPdg[j] == 211 && fKE >= pion_ke) ||
        (tStdHepPdg[j] == -211 && fKE >= pion_ke) ||
        (tStdHepPdg[j] == 111) ||
        ((tStdHepPdg[j] == 321 || tStdHepPdg[j] == -321 || tStdHepPdg[j] == 130 || tStdHepPdg[j] == 310) && fKE >= kaon_ke) ||
        ((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE >= muon_ke) ||
        ((tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE >= electron_ke) ||
        (tStdHepPdg[j] == 22 && fKE_gamma >= 2 * electron_ke))
    {

        if (tStdHepPdg[j] == 22)
        {
            // std::cout << "fKE_gamma: " << fKE_gamma << std::endl;
            // std::cout << "fAbsoluteParticleMomentum_gamma " << fAbsoluteParticleMomentum_gamma << std::endl;
            // std::cout << "tStdHepP4[4 * j + 3]: " << tStdHepP4[4 * j + 3] << std::endl;
            fInvMass = 0.;
        }

        /* pdgs.push_back(tStdHepPdg[j]);
         masses.push_back(fInvMass);
         energies.push_back(fKE);
         pxs.push_back(tStdHepP4[4 * j]);
         pys.push_back(tStdHepP4[4 * j + 1]);
         pzs.push_back(tStdHepP4[4 * j + 2]);
         costheta_arr.push_back(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
         theta_arr.push_back((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));

         tot_fKE += fKE;
         tot_fpx += tStdHepP4[4 * j];
         tot_fpy += tStdHepP4[4 * j + 1];
         tot_fpz += tStdHepP4[4 * j + 2];

         if (tStdHepPdg[j] == 2212)
         {
             Fin_Prot_Mom->Fill(1000. * fAbsoluteParticleMomentum);
         }
         if (tStdHepPdg[j] == 22)
         {
             Fin_Gamma_Mom->Fill(1000. * fAbsoluteParticleMomentum_gamma);
         }
         if (tStdHepPdg[j] == 211)
         {
             Fin_PiPlus_Mom->Fill(1000. * fAbsoluteParticleMomentum);
         }

         if (tStdHepPdg[j] == -211)
         {
             Fin_PiMinus_Mom->Fill(1000. * fAbsoluteParticleMomentum);
         }

         if (tStdHepPdg[j] == 111)
         {
             Fin_PiZero_Mom->Fill(1000. * fAbsoluteParticleMomentum);
         }
         */
        // ─── split photon vs everything else ───
        if (tStdHepPdg[j] == 22)
        {
            // PHOTONS: zero mass + massless kinematics everywhere
            pdgs.push_back(22);
            masses.push_back(0.0);
            energies.push_back(fKE_gamma);
            pxs.push_back(tStdHepP4[4 * j]);
            pys.push_back(tStdHepP4[4 * j + 1]);
            pzs.push_back(tStdHepP4[4 * j + 2]);

            costheta_arr.push_back(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum_gamma);
            theta_arr.push_back((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum_gamma));

            tot_fKE += fKE_gamma;
            tot_fpx += tStdHepP4[4 * j];
            tot_fpy += tStdHepP4[4 * j + 1];
            tot_fpz += tStdHepP4[4 * j + 2];

            Fin_Gamma_Mom->Fill(1000. * fAbsoluteParticleMomentum_gamma);
        }
        else
        {
            // ALL OTHER PARTICLES: keep original (massive) kinematics
            pdgs.push_back(tStdHepPdg[j]);
            masses.push_back(fInvMass);
            energies.push_back(fKE);
            pxs.push_back(tStdHepP4[4 * j]);
            pys.push_back(tStdHepP4[4 * j + 1]);
            pzs.push_back(tStdHepP4[4 * j + 2]);

            costheta_arr.push_back(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
            theta_arr.push_back((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));

            tot_fKE += fKE;
            tot_fpx += tStdHepP4[4 * j];
            tot_fpy += tStdHepP4[4 * j + 1];
            tot_fpz += tStdHepP4[4 * j + 2];

            if (tStdHepPdg[j] == 2212)
                Fin_Prot_Mom->Fill(1000. * fAbsoluteParticleMomentum);
            if (tStdHepPdg[j] == 211)
                Fin_PiPlus_Mom->Fill(1000. * fAbsoluteParticleMomentum);
            if (tStdHepPdg[j] == -211)
                Fin_PiMinus_Mom->Fill(1000. * fAbsoluteParticleMomentum);
            if (tStdHepPdg[j] == 111)
                Fin_PiZero_Mom->Fill(1000. * fAbsoluteParticleMomentum);

            if ((tStdHepPdg[j] == 2212 && fKE >= prot_ke) ||
                (tStdHepPdg[j] == 211 && fKE >= pion_ke) ||
                (tStdHepPdg[j] == -211 && fKE >= pion_ke) ||
                (tStdHepPdg[j] == 111) ||
                ((tStdHepPdg[j] == 321 || tStdHepPdg[j] == -321 || tStdHepPdg[j] == 130 || tStdHepPdg[j] == 310) && fKE >= kaon_ke))
            {
                tot_hadronic_energy += fKE;
            }
        }
    }
}

void ScalarLept_wNC(const std::string &input_file)
{
    // Set a maximum event particle list length
    const int NMaxParticlesPerEvent = 1000;

    std::map<std::string, std::variant<int, float, std::string>> dictionary = process_file(input_file);

    // Load ROOT file (example usage)
    std::string Input_Root_file = std::get<std::string>(dictionary["INPUT_ROOT_FILE"]);
    TFile *SignalFile = TFile::Open(Input_Root_file.c_str(), "READ");
    TTree *SignalTree = (TTree *)SignalFile->Get("gRooTracker");

    int NumberEntries = SignalTree->GetEntries();

    // Handle Num_events
    auto Num_events_var = dictionary["EventNum"];
    int Num_events;

    if (std::holds_alternative<std::string>(Num_events_var) && std::get<std::string>(Num_events_var) == "All")
    {
        Num_events = NumberEntries;
    }
    else if (std::holds_alternative<int>(Num_events_var))
    {
        Num_events = std::get<int>(Num_events_var);
        if (Num_events > NumberEntries)
        {
            // std::cout << "Max reached" << std::endl;
            Num_events = NumberEntries;
        }
    }

    std::vector<std::variant<int, std::string>> Init_PDG_arr;
    const auto &Init_PDG_Values = dictionary.at("Initial_PDG");

    if (std::holds_alternative<int>(Init_PDG_Values))
    {
        Init_PDG_arr.push_back(std::get<int>(Init_PDG_Values));
    }
    else if (std::holds_alternative<std::string>(Init_PDG_Values))
    {
        const std::string &str_value = std::get<std::string>(Init_PDG_Values);
        if (str_value == "Any")
        {
            Init_PDG_arr.push_back("Any");
        }
        else
        {
            // Split by commas and add each token as an integer
            std::vector<std::string> tokens = split(str_value, ',');
            for (const std::string &token : tokens)
            {
                Init_PDG_arr.push_back(std::stoi(token));
            }
        }
    }

    std::variant<int, float, std::string> num_proton_value;
    std::variant<int, float, std::string> num_pion_value;
    std::variant<int, float, std::string> Final_lepton_PDG;

    double Min_Nu_energy = getDoubleValue(dictionary.at("Initial_Nu_Energy_min"));
    double Max_Nu_energy = getDoubleValue(dictionary.at("Initial_Nu_Energy_max"));

    getValue(dictionary, "num_proton", num_proton_value);
    getValue(dictionary, "num_pion", num_pion_value);
    getValue(dictionary, "Final_state_lepton_PDG_code_or_process", Final_lepton_PDG);

    int int_num_prot = -1;
    std::string str_num_prot;
    bool is_num_prot_int = false;

    if (std::holds_alternative<int>(num_proton_value))
    {
        int_num_prot = std::get<int>(num_proton_value);
        is_num_prot_int = true;
    }
    else if (std::holds_alternative<std::string>(num_proton_value))
    {
        str_num_prot = std::get<std::string>(num_proton_value);
    }

    int int_num_pion = -1;
    std::string str_num_pion;
    bool is_num_pion_int = false;

    if (std::holds_alternative<int>(num_pion_value))
    {
        int_num_pion = std::get<int>(num_pion_value);
        is_num_pion_int = true;
    }
    else if (std::holds_alternative<std::string>(num_pion_value))
    {
        str_num_pion = std::get<std::string>(num_pion_value);
    }

    int int_Final_lepton_PDG = -1;
    std::string str_Final_lepton_PDG;
    bool is_Final_lepton_PDG_int = false;

    if (std::holds_alternative<int>(Final_lepton_PDG))
    {
        int_Final_lepton_PDG = std::get<int>(Final_lepton_PDG);
        is_Final_lepton_PDG_int = true;
    }
    else if (std::holds_alternative<std::string>(Final_lepton_PDG))
    {
        str_Final_lepton_PDG = std::get<std::string>(Final_lepton_PDG);
    }

    std::string num_events_str = variantToString(dictionary["EventNum"]);
    std::string proton_num_str = variantToString(dictionary["num_proton"]);
    std::string pion_num_str = variantToString(dictionary["num_pion"]);

   
     // directory here

    std::string output_directory = std::get<std::string>(dictionary["OUTPUT_DIR"]);
    std::string directory = output_directory + "ScalarLeptwNC_eventnum_" + num_events_str + "_" + proton_num_str + "p" + pion_num_str + "pi";

    try
    {
        if (std::filesystem::create_directories(directory))
        {
            std::cout << "Directory created successfully: " << directory << std::endl;
        }
        else
        {
            std::cout << "Directory already exists or could not be created: " << directory << std::endl;
        }
    }
    catch (const std::filesystem::filesystem_error &e)
    {
        std::cerr << "An error occurred while creating the directory: " << e.what() << std::endl;
    }

    std::string Output_Root_file = output_directory + "ScalarLeptwNC_eventnum_" + num_events_str + "_" + proton_num_str + "p" + pion_num_str + "pi.root";

    TFile *treefile = new TFile(Output_Root_file.c_str(), "RECREATE");

    std::string output_name = std::get<std::string>(dictionary["OUTPUT_NAME"]);
    std::string outfile_name = output_directory + "ScalarLeptwNC_eventnum_" + num_events_str + "_" + proton_num_str + "p" + pion_num_str + "pi.csv";

    ofstream outfile(outfile_name);
    std::string last_name = output_name + "ScalarLeptwNC_eventnum_"+ num_events_str+ "_" + proton_num_str + "p" + pion_num_str + "pi_";

    // define the header for the CSV file
    outfile << "\"Event_Index\",\"Nu_PDG\",\"Nu_Energy\",\"Nu_Mom_X\",\"Nu_Mom_Y\",\"Nu_Mom_Z\",\"Nu_CosTheta\",\"Nu_Theta\",\"Nu_Phi\",\"Nu_Theta_z\",\"Nu_Phi_z\",\"Nu_Baseline\",\"Lept_PDG\",\"Lept_Mass\",\"Lept_Energy\",\"Lept_MomX\",\"Lept_MomY\",\"Lept_MomZ\",\"Lept_CosTheta\",\"Lept_Theta\",\"Lept_Theta_z\",\"Lept_Phi_z\",\"Final_State_Particles_PDG\",\"Final_State_Particles_Mass\",\"Final_State_Particles_Energy\",\"Final_State_Particles_Momentum_X\",\"Final_State_Particles_Momentum_Y\",\"Final_State_Particles_Momentum_Z\",\"Final_State_Particles_CosTheta\",\"Final_State_Particles_Theta\",\"tot_fKE\",\"tot_hadronic_energy\",\"p_tot\",\"P_miss\",\"MissE\",\"P_miss_x\",\"P_miss_y\",\"P_miss_z\",\"Topology\"\n";
    // outfile << "\"Event_Index\",\"Initial_State_Neutrino_PDG\",\"Initial_State_Neutrino_Energy\",\"Initial_State_Neutrino_Momentum_X\",\"Initial_State_Neutrino_Momentum_Y\",\"Initial_State_Neutrino_Momentum_Z\",\"Initial_Neutrino_CosTheta\",\"Initial_Neutrino_Theta\",\"Final_State_Particles_PDG\",\"Final_State_Particles_Mass\",\"Final_State_Particles_Energy\",\"Final_State_Particles_Momentum_X\",\"Final_State_Particles_Momentum_Y\",\"Final_State_Particles_Momentum_Z\",\"Final_State_Particles_CosTheta\",\"Final_State_Particles_Theta\",\"tot_fKE\",\"p_tot\",\"P_miss\"\n";



    gStyle->SetStatY(0.9);
    gStyle->SetStatX(0.9);

    TCanvas *c1 = new TCanvas("canvas", "canvas", 1500, 1000);

    Init_Nu_Mom = new TH1D((last_name + "Init_Nu_Mom").c_str(), (last_name + "Initial Neutrino Momentum, AR23").c_str(), 1000, 100., 20000.);
    Init_Nu_Mom->GetXaxis()->SetTitle("Initial Neutrino Momentum [MeV/c]");
    Init_Nu_Mom->GetYaxis()->SetTitle("Counts");
    Init_Nu_Mom->SetLineColor(kGreen);
    Init_Nu_Mom->SetLineWidth(3);

    // Initial neutrino angle with respect to zenith (y-axis in DUNE geometry/coordinate system)
    Init_Nu_Theta = new TH1D((last_name + "Init_Nu_Theta").c_str(), (last_name + "Initial Neutrino #theta, AR23").c_str(), 1000, 0., 0.);
    Init_Nu_Theta->GetXaxis()->SetTitle("Initial Neutrino #theta [deg]");
    Init_Nu_Theta->GetYaxis()->SetTitle("Counts");
    Init_Nu_Theta->SetLineColor(kGreen);
    Init_Nu_Theta->SetLineWidth(3);

    Init_Nu_CosTheta = new TH1D((last_name + "Init_Nu_CosTheta").c_str(), (last_name + "Initial Neutrino cos(#theta), AR23").c_str(), 1000, 0., 0.); //,16.0);
    Init_Nu_CosTheta->GetXaxis()->SetTitle("Initial Neutrino Cos(#theta)");
    Init_Nu_CosTheta->GetYaxis()->SetTitle("Counts");
    Init_Nu_CosTheta->SetLineColor(kGreen);
    Init_Nu_CosTheta->SetLineWidth(3);

    // for phi_nu
    Init_Nu_Phi = new TH1D((last_name + "Init_Nu_Phi").c_str(), (last_name + "Initial Neutrino #phi, AR23").c_str(), 1000, 0., 0.);
    Init_Nu_Phi->GetXaxis()->SetTitle("Initial Neutrino #phi[deg])");
    Init_Nu_Phi->GetYaxis()->SetTitle("Counts");
    Init_Nu_Phi->SetLineColor(kGreen);
    Init_Nu_Phi->SetLineWidth(3);

    // Make oscillogram plot
    Oscillogram = new TH2D((last_name + "Oscillogram_hA_2018").c_str(), (last_name + "Oscillogram, AR23").c_str(), 1000, 0., 0., 1000, 0., 0.);
    Oscillogram->GetXaxis()->SetTitle("cos(#theta_{#nu}) from Homestake Zenith");
    Oscillogram->GetYaxis()->SetTitle("Initial Neutrino Energy [GeV]");

    Fin_CC_Lept_Mom = new TH1D((last_name + "Fin_CC_Lept_Mom").c_str(), (last_name + "Final State CC-Produced Lepton Momentum, AR23").c_str(), 1000, 0., 1500.);
    Fin_CC_Lept_Mom->GetXaxis()->SetTitle("Final State CC-Produced Lepton Momentum [MeV/c]");
    Fin_CC_Lept_Mom->GetYaxis()->SetTitle("Counts");
    Fin_CC_Lept_Mom->SetLineColor(kViolet);
    Fin_CC_Lept_Mom->SetLineWidth(3);

    Fin_CC_Lept_Theta = new TH1D((last_name + "Fin_CC_Lept_Theta").c_str(), (last_name + "Final CC Lepton #theta, AR23").c_str(), 1000, 0., 0.);
    Fin_CC_Lept_Theta->GetXaxis()->SetTitle("Final CC Lepton #theta [deg]");
    Fin_CC_Lept_Theta->GetYaxis()->SetTitle("Counts");
    Fin_CC_Lept_Theta->SetLineColor(kGreen);
    Fin_CC_Lept_Theta->SetLineWidth(3);

    Fin_CC_Lept_CosTheta = new TH1D((last_name + "Fin_CC_Lept_CosTheta").c_str(), (last_name + "Final CC Lepton cos(#theta), AR23").c_str(), 1000, 0., 0.);
    Fin_CC_Lept_CosTheta->GetXaxis()->SetTitle("Final CC Lepton Cos(#theta)");
    Fin_CC_Lept_CosTheta->GetYaxis()->SetTitle("Counts");
    Fin_CC_Lept_CosTheta->SetLineColor(kGreen);
    Fin_CC_Lept_CosTheta->SetLineWidth(3);

    Fin_NC_Lept_Mom = new TH1D((last_name + "Fin_NC_Lept_Mom").c_str(), (last_name + "Final State NC-Produced Lepton Momentum, AR23").c_str(), 1000, 0., 1500.);
    Fin_NC_Lept_Mom->GetXaxis()->SetTitle("Final State NC-Produced Lepton Momentum [MeV/c]");
    Fin_NC_Lept_Mom->GetYaxis()->SetTitle("Counts");
    Fin_NC_Lept_Mom->SetLineColor(kViolet);
    Fin_NC_Lept_Mom->SetLineWidth(3);

    Fin_NC_Lept_Theta = new TH1D((last_name + "Fin_NC_Lept_Theta").c_str(), (last_name + "Final NC Lepton #theta, AR23").c_str(), 1000, 0., 0.);
    Fin_NC_Lept_Theta->GetXaxis()->SetTitle("Final NC Lepton #theta [deg]");
    Fin_NC_Lept_Theta->GetYaxis()->SetTitle("Counts");
    Fin_NC_Lept_Theta->SetLineColor(kGreen);
    Fin_NC_Lept_Theta->SetLineWidth(3);

    Fin_NC_Lept_CosTheta = new TH1D((last_name + "Fin_NC_Lept_CosTheta").c_str(), (last_name + "Final NC Lepton cos(#theta), AR23").c_str(), 1000, 0., 0.);
    Fin_NC_Lept_CosTheta->GetXaxis()->SetTitle("Final NC Lepton Cos(#theta)");
    Fin_NC_Lept_CosTheta->GetYaxis()->SetTitle("Counts");
    Fin_NC_Lept_CosTheta->SetLineColor(kGreen);
    Fin_NC_Lept_CosTheta->SetLineWidth(3);

    Fin_Prot_Mom = new TH1D((last_name + "Fin_Prot_Mom").c_str(), (last_name + "Final State Proton Momentum, AR23").c_str(), 1000, 0., 1500.);

    // Fin_Prot_Mom = new TH1D((last_name + "Fin_Prot_Mom_low").c_str(), (last_name + "Final State Proton Momentum_low, AR23").c_str(), 1000, -100, 500);
    Fin_Prot_Mom->GetXaxis()->SetTitle("Final State Proton Momentum [MeV/c]");
    Fin_Prot_Mom->GetYaxis()->SetTitle("Counts");
    Fin_Prot_Mom->SetLineColor(kRed);
    Fin_Prot_Mom->SetLineWidth(3);

    Fin_PiPlus_Mom = new TH1D((last_name + "Fin_PiPlus_Mom").c_str(), (last_name + "Final State #pi^{+} Momentum, AR23").c_str(), 1000, 0., 1500.);
    // Fin_PiPlus_Mom = new TH1D((last_name + "Fin_PiPlus_Mom_low").c_str(), (last_name + "Final State #pi^{+} Momentum_low, AR23").c_str(), 1000, -100, 500);
    Fin_PiPlus_Mom->GetXaxis()->SetTitle("Final State #pi^{+} Momentum [MeV/c]");
    Fin_PiPlus_Mom->GetYaxis()->SetTitle("Counts");
    Fin_PiPlus_Mom->SetLineColor(kRed + 1);
    Fin_PiPlus_Mom->SetLineWidth(3);

    Fin_PiMinus_Mom = new TH1D((last_name + "Fin_PiMinus_Mom").c_str(), (last_name + "Final State #pi^{-} Momentum, AR23").c_str(), 1000, 0., 1500.);
    // Fin_PiMinus_Mom = new TH1D((last_name + "Fin_PiMinus_Mom_low").c_str(), (last_name + "Final State #pi^{-} Momentum_Low, AR23").c_str(), 1000, -100, 500);
    Fin_PiMinus_Mom->GetXaxis()->SetTitle("Final State #pi^{-} Momentum [MeV/c]");
    Fin_PiMinus_Mom->GetYaxis()->SetTitle("Counts");
    Fin_PiMinus_Mom->SetLineColor(kBlue + 1);
    Fin_PiMinus_Mom->SetLineWidth(3);

    Fin_PiZero_Mom = new TH1D((last_name + "Fin_PiZero_Mom").c_str(), (last_name + "Final State #pi^{0} Momentum, AR23").c_str(), 1000, 0., 1500.);
    // Fin_PiZero_Mom = new TH1D((last_name + "Fin_PiZero_Mom_low").c_str(), (last_name + "Final State #pi^{0} Momentum_Low, AR23").c_str(), 1000, -100,500);
    Fin_PiZero_Mom->GetXaxis()->SetTitle("Final State #pi^{0} Momentum [MeV/c]");
    Fin_PiZero_Mom->GetYaxis()->SetTitle("Counts");
    Fin_PiZero_Mom->SetLineColor(kGreen + 1);
    Fin_PiZero_Mom->SetLineWidth(3);

    Fin_Prot_Mult = new TH1D((last_name + "Fin_Prot_Mult").c_str(), (last_name + "Final State Proton Multiplicity, AR23").c_str(), 20, -0.5, 19.5);
    Fin_Prot_Mult->GetXaxis()->SetTitle("Final State Proton Multiplicity");
    Fin_Prot_Mult->GetYaxis()->SetTitle("Events");
    Fin_Prot_Mult->SetLineColor(kRed);
    Fin_Prot_Mult->SetLineWidth(3);

    Fin_PiPlus_Mult = new TH1D((last_name + "Fin_PiPlus_Mult").c_str(), (last_name + "Final State #pi^{+} Multiplicity, AR23").c_str(), 7, -0.5, 6.5);
    Fin_PiPlus_Mult->GetXaxis()->SetTitle("Final State #pi^{+} Multiplicity");
    Fin_PiPlus_Mult->GetYaxis()->SetTitle("Events");
    Fin_PiPlus_Mult->SetLineColor(kRed + 1);
    Fin_PiPlus_Mult->SetLineWidth(3);

    Fin_PiMinus_Mult = new TH1D((last_name + "Fin_PiMinus_Mult").c_str(), (last_name + "Final State #pi^{-} Multiplicity, AR23").c_str(), 7, -0.5, 6.5);
    Fin_PiMinus_Mult->GetXaxis()->SetTitle("Final State #pi^{-} Multiplicity");
    Fin_PiMinus_Mult->GetYaxis()->SetTitle("Events");
    Fin_PiMinus_Mult->SetLineColor(kBlue + 1);
    Fin_PiMinus_Mult->SetLineWidth(3);

    Fin_PiZero_Mult = new TH1D((last_name + "Fin_PiZero_Mult").c_str(), (last_name + "Final State #pi^{0} Multiplicity, AR23").c_str(), 7, -0.5, 6.5);
    Fin_PiZero_Mult->GetXaxis()->SetTitle("Final State #pi^{0} Multiplicity");
    Fin_PiZero_Mult->GetYaxis()->SetTitle("Events");
    Fin_PiZero_Mult->SetLineColor(kGreen + 1);
    Fin_PiZero_Mult->SetLineWidth(3);

    Fin_Gamma_Mom = new TH1D((last_name + "Fin_Gamma_Mom").c_str(), (last_name + "Final State #gamma Momentum, AR23").c_str(), 1000, 0., 500.);
    Fin_Gamma_Mom->GetXaxis()->SetTitle("Final State #gamma Momentum [MeV/c]");
    Fin_Gamma_Mom->GetYaxis()->SetTitle("Counts");
    Fin_Gamma_Mom->SetLineColor(kBlue + 1);
    Fin_Gamma_Mom->SetLineWidth(3);

    int tStdHepN = 0;                               // the num of particles in an event
    int tStdHepStatus[NMaxParticlesPerEvent] = {0}; // an array with the all the number of elements in the tStdHepStatus set to 0
    int tStdHepPdg[NMaxParticlesPerEvent] = {0};
    double tStdHepP4[4 * NMaxParticlesPerEvent] = {0.};
    double tStdHepX4[4 * NMaxParticlesPerEvent] = {0.};

    SignalTree->SetBranchAddress("StdHepN", &tStdHepN);
    SignalTree->SetBranchAddress("StdHepStatus", &tStdHepStatus);
    SignalTree->SetBranchAddress("StdHepPdg", &tStdHepPdg);
    SignalTree->SetBranchAddress("StdHepP4", &tStdHepP4);
    SignalTree->SetBranchAddress("StdHepX4", &tStdHepX4);

    std::vector<double> energies, masses, pxs, pys, pzs, costheta_arr, theta_arr;
    std::vector<int> pdgs;

    // Reading different kinetic energies from the dictionary
    double muon_ke = getDoubleValue(dictionary.at("Muon_KE"));
    double electron_ke = getDoubleValue(dictionary.at("Electron_KE"));
    double prot_ke = getDoubleValue(dictionary.at("Proton_KE"));
    double kaon_ke = getDoubleValue(dictionary.at("K+-_KE"));
    double pion_ke = getDoubleValue(dictionary.at("Pi+-_KE"));

    //////////////START OF THE EVENT LOOP//////
    for (int i = 0; i < Num_events; i++)
    { // Initializing multiplicity variables
        SignalTree->GetEntry(i);

        if ((tStdHepPdg[4] == 22 && abs(tStdHepPdg[0] - tStdHepPdg[5]) > 1) || (tStdHepPdg[4] != 22 && abs(tStdHepPdg[0] - tStdHepPdg[4]) > 1))
            continue;

        int n_prot = 0, n_piplus = 0, n_piminus = 0, n_pizero = 0, n_pi = 0, visible_count = 0;
        int N_GAMMAS;

        energies.clear();
        masses.clear();
        pxs.clear();
        pys.clear();
        pzs.clear();
        costheta_arr.clear();
        theta_arr.clear();
        pdgs.clear();

        double tot_fpx = 0, tot_fpy = 0, tot_fpz = 0;
        double tot_fKE = 0;
        double tot_hadronic_energy = 0;
        double nu_energy = 0, nu_px = 0, nu_py = 0, nu_pz = 0;
        double lep_energy = 0, lep_px = 0, lep_py = 0, lep_pz = 0;

        bool skip_event = false;

        // Print out the current event number to screen
        //  //std::cout << "The current event number is " << i << " of " << NumberEntries << endl;
        // Get the event entry information

        if (i % 100000 == 0)
        {
            std::cout << "The current event number in the 2nd event loop is " << i << " of " << NumberEntries << endl;
        }
        int tStdHepPdg_nu, lepton_pdg_instore;
        double fAbsoluteParticleMomentum_nu, fKE_nu, fAbsoluteParticleMomentum_lept, fKE_lept;

        for (int j = 0; j < tStdHepN; j++)
        {

            if (tStdHepStatus[j] == 0 && (tStdHepPdg[j] == -16 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16))
            {
                tStdHepPdg_nu = tStdHepPdg[j];
                auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);
                fAbsoluteParticleMomentum_nu = fAbsoluteParticleMomentum;
                fKE_nu = fKE;
            }

            if (tStdHepStatus[j] == 1)
            {
                if ((tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11 || tStdHepPdg[j] == 12 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == 15 || tStdHepPdg[j] == -15 || tStdHepPdg[j] == 16 || tStdHepPdg[j] == -16) && j <= 5)
                {
                    lepton_pdg_instore = tStdHepPdg[j];
                    auto [fAbsoluteParticleMomentum, fInvMass, fKE] = kinematics(tStdHepP4, j);
                    fAbsoluteParticleMomentum_lept = fAbsoluteParticleMomentum;
                    fKE_lept = fKE;
                    if ((tStdHepPdg[j] == 13 && fKE_lept >= muon_ke) || (tStdHepPdg[j] == -13 && fKE_lept >= muon_ke) || (tStdHepPdg[j] == 11 && fKE_lept >= electron_ke) || (tStdHepPdg[j] == -11 && fKE_lept >= electron_ke))
                    {
                        visible_count++;
                    }
                }

                if (tStdHepPdg[j] == 2212)
                {
                    auto [fAbsoluteParticleMomentum, fInvMass, fKE] = kinematics(tStdHepP4, j);
                    if (fAbsoluteParticleMomentum <= 0.)
                    {
                        skip_event = true;
                        break;
                    }
                    if (fKE >= prot_ke)
                    {
                        visible_count++;
                        n_prot++;
                    }
                }
                if (tStdHepPdg[j] == 211 || tStdHepPdg[j] == -211)
                {
                    auto [fAbsoluteParticleMomentum, fInvMass, fKE] = kinematics(tStdHepP4, j);
                    // fAbsoluteParticleMomentum_pion=fAbsoluteParticleMomentum;
                    if (fAbsoluteParticleMomentum <= 0.)
                    {
                        skip_event = true;
                        break;
                    }
                    if (fKE >= pion_ke)
                    {
                        visible_count++;
                        n_pi++;

                        if (tStdHepPdg[j] == 211)
                        {
                            n_piplus++;
                        }
                        else if (tStdHepPdg[j] == -211)
                        {
                            n_piminus++;
                        }
                    }
                }

                if (tStdHepPdg[j] == 111)
                {
                    visible_count++, n_pi++, n_pizero++;
                }

                if (tStdHepPdg[j] == 321 || tStdHepPdg[j] == -321 || tStdHepPdg[j] == 130 || tStdHepPdg[j] == 310)
                {
                    auto [fAbsoluteParticleMomentum, fInvMass, fKE] = kinematics(tStdHepP4, j);
                    if (fAbsoluteParticleMomentum <= 0.)
                    {
                        skip_event = true;
                        break;
                    }
                    // fAbsoluteParticleMomentum_kaon=fAbsoluteParticleMomentum;
                    if (fKE >= kaon_ke)
                    {
                        visible_count++;
                    }
                }
                if (tStdHepPdg[j] == 22)
                {
                    auto [fAbsoluteParticleMomentum_gamma, fKE_gamma] = kinematics_massless(tStdHepP4, j);
                    if (fAbsoluteParticleMomentum_gamma <= 0.)
                    {
                        skip_event = true;
                        std::cout << "About to skip event because of bad photon" << std::endl;
                        break;
                    }
                    if (fKE_gamma >= 2 * electron_ke)
                    {
                        visible_count++;
                    }
                    N_GAMMAS++;
                    // std::cout << "N_GAMMA: " << N_GAMMAS << std::endl;
                    // std::cout << "     fKE_gamma inside if statement with invisible gammas" << fKE_gamma << std::endl;
                }
            }
        }
        int init_lep_code = 0;
        if (tStdHepPdg_nu == 12)
            init_lep_code = 1; // nue
        else if (tStdHepPdg_nu == -12)
            init_lep_code = 2; // nuebar
        else if (tStdHepPdg_nu == 14)
            init_lep_code = 3; // numu
        else if (tStdHepPdg_nu == -14)
            init_lep_code = 4; // numubar
        else if (tStdHepPdg_nu == 16)
            init_lep_code = 5; // nutau
        else if (tStdHepPdg_nu == -16)
            init_lep_code = 6; // nutaubar
        std::ostringstream topo_ss;
        topo_ss << init_lep_code
                << std::setw(2) << std::setfill('0') << n_prot << "00"
                << std::setw(2) << std::setfill('0') << n_piplus << "00"
                << std::setw(2) << std::setfill('0') << n_piminus << "00"
                << std::setw(2) << std::setfill('0') << n_pizero;
        std::string topology = topo_ss.str();
        if (visible_count == 0 || skip_event)
        {
            // //std::cout << "I am skipping for i: " << i << endl;
            continue;
        }

        bool found = false;
        bool string_found = false;
        for (const auto &value : Init_PDG_arr)
        {
            if (std::holds_alternative<int>(value) && std::get<int>(value) == tStdHepPdg_nu)
            {
                found = true;
                break;
            }

            else if (std::holds_alternative<std::string>(value))
            {
                string_found = true;
                break;
            }
        }

        // if (found) {
        //     //std::cout << "tStdHepPdg_nu is in Init_PDG_arr" <<"for i  "<<i<< std::endl;
        // } else {
        //     //std::cout << "tStdHepPdg_nu is not in Init_PDG_arr" << "for i  "<<i<<std::endl;
        // }

        // SpecificNu and Specific NC/CC

        if (found && (is_Final_lepton_PDG_int && int_Final_lepton_PDG == lepton_pdg_instore) && Min_Nu_energy < fKE_nu && fKE_nu < Max_Nu_energy &&
            ((is_num_prot_int && is_num_pion_int && n_prot == int_num_prot && n_pi == int_num_pion) ||
             (!is_num_prot_int && str_num_prot == "N" && is_num_pion_int && n_pi == int_num_pion) ||
             (is_num_prot_int && n_prot == int_num_prot && !is_num_pion_int && str_num_pion == "N") ||
             (!is_num_prot_int && str_num_prot == "N" && !is_num_pion_int && str_num_pion == "N")))
        {

            // std::cout << "I am here yay for i  "<<i<< std::endl;

            for (int j = 0; j < tStdHepN; j++)
            {
                // //std::cout <<" i issss "<<i<< " and j isssss " << j << " and status is " << tStdHepStatus[j] <<" and pdg " << tStdHepPdg[j]<< std::endl;

                if (tStdHepStatus[j] == 0 && (tStdHepPdg[j] == -16 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16))
                {

                    auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);

                    nu_energy = fKE;
                    nu_px = tStdHepP4[4 * j];
                    nu_py = tStdHepP4[4 * j + 1];
                    nu_pz = tStdHepP4[4 * j + 2];

                    double baseline = calc_baseline(tStdHepP4, fAbsoluteParticleMomentum, j);
                    double phi = phi_nu(tStdHepP4, fAbsoluteParticleMomentum, j);

                    Init_Nu_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                    Init_Nu_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    Init_Nu_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    Init_Nu_Phi->Fill(phi);
                    Oscillogram->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum, fAbsoluteParticleMomentum);

                    // outfile << "\"" << i << "\",";
                    double theta_z_nu = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    double phi_z_nu = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                    outfile << "\"" << i << "\"," << std::setprecision(6) << "\"" << tStdHepPdg[j] << "\",\"" << fAbsoluteParticleMomentum << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << phi << "\",\"" << theta_z_nu << "\",\"" << phi_z_nu << "\",\"" << baseline << "\",\"";

                    // Output the results
                    // //std::cout << "i AM HEre for i:"<<i<<"and j:"<< j << std::endl;

                    // //std::cout << "Absolute Particle Momentum: " << fAbsoluteParticleMomentum << std::endl;
                    // // //std::cout << "Invariant Mass: " << fInvMass << std::endl;
                    // //std::cout << "Kinetic Energy: " << fKE << std::endl;
                }

                if (tStdHepStatus[j] == 1)
                {
                    // Charged current cases

                    auto [fAbsoluteParticleMomentum, fInvMass, fKE] = kinematics(tStdHepP4, j);

                    if (((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE >= muon_ke ||
                         (tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE >= electron_ke ||
                         (tStdHepPdg[j] == 15 || tStdHepPdg[j] == -15)) &&
                        j <= 5)
                    {

                        double theta_z_lep = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        double phi_z_lep = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                        outfile << std::setprecision(6) << tStdHepPdg[j] << "\",\"" << fInvMass << "\",\"" << fKE << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << theta_z_lep << "\",\"" << phi_z_lep << "\",\"";

                        lep_energy = fKE;
                        lep_px = tStdHepP4[4 * j];
                        lep_py = tStdHepP4[4 * j + 1];
                        lep_pz = tStdHepP4[4 * j + 2];

                        tot_fKE += fKE;
                        tot_fpx += tStdHepP4[4 * j];
                        tot_fpy += tStdHepP4[4 * j + 1];
                        tot_fpz += tStdHepP4[4 * j + 2];

                        Fin_CC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                        Fin_CC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        Fin_CC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    }

                    else if (((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE < muon_ke ||
                              (tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE < electron_ke) &&
                             j <= 5)
                    {

                        outfile << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"";
                    }

                    else if ((tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -16) && j <= 5)
                    {
                        auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);
                        double theta_z_lep = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        double phi_z_lep = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                        outfile << std::setprecision(6) << tStdHepPdg[j] << "\",\"" << 0 << "\",\"" << fKE << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << theta_z_lep << "\",\"" << phi_z_lep << "\",\"";

                        Fin_NC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                        Fin_NC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        Fin_NC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    }

                    else if (tStdHepPdg[j] == 2212 || tStdHepPdg[j] == 211 || tStdHepPdg[j] == -211 || tStdHepPdg[j] == 111 || tStdHepPdg[j] == 321 || tStdHepPdg[j] == -321 || tStdHepPdg[j] == 130 || tStdHepPdg[j] == 310 || tStdHepPdg[j] == 22 || tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11 || tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13)
                    {
                        finalparticles_info(tStdHepP4, j, tStdHepPdg, pdgs, masses,
                                            energies, pxs, pys, pzs, costheta_arr, theta_arr, tot_fKE, tot_hadronic_energy, tot_fpx, tot_fpy, tot_fpz, dictionary);
                    }
                }
            }

            // //do_resize(pdgs,masses,energies,pxs,pys,pzs,costheta_arr,theta_arr);

            Fin_Prot_Mult->Fill(n_prot);
            n_prot = 0;
            Fin_PiPlus_Mult->Fill(n_piplus);
            n_piplus = 0;
            Fin_PiMinus_Mult->Fill(n_piminus);
            n_piminus = 0;
            Fin_PiZero_Mult->Fill(n_pizero);
            n_pizero = 0;

            if (pdgs.empty())
            {
                pdgs.push_back(0);
            }

            if (!pdgs.empty())
            {
                for (int j = 0; j < pdgs.size(); j++)
                {
                    if (j < pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << ",";
                    }
                    if (j == pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << "\"";
                    }
                }
            }
            outfile << ",\"";

            // Check if each array is empty, and if so, replace it with a single 0
            if (masses.empty())
                masses.push_back(0);
            if (energies.empty())
                energies.push_back(0);
            if (pxs.empty())
                pxs.push_back(0);
            if (pys.empty())
                pys.push_back(0);
            if (pzs.empty())
                pzs.push_back(0);
            if (costheta_arr.empty())
                costheta_arr.push_back(0);
            if (theta_arr.empty())
                theta_arr.push_back(0);

            // Now write each array to the file
            writeVectorToFile(outfile, masses);
            writeVectorToFile(outfile, energies);
            writeVectorToFile(outfile, pxs);
            writeVectorToFile(outfile, pys);
            writeVectorToFile(outfile, pzs);
            writeVectorToFile(outfile, costheta_arr);
            writeVectorToFile(outfile, theta_arr);

            double p_tot = sqrt(pow(tot_fpx, 2) + pow(tot_fpy, 2) + pow(tot_fpz, 2));
            double p_miss = tot_fKE - p_tot;
            double MissE = nu_energy - tot_fKE;
            double P_miss_x = nu_px - tot_fpx;
            double P_miss_y = nu_py - tot_fpy;
            double P_miss_z = nu_pz - tot_fpz;

            double P_miss_magnitude = sqrt(pow(P_miss_x, 2) + pow(P_miss_y, 2) + pow(P_miss_z, 2));
            double theta_z = (P_miss_magnitude > 0) ? (180. / PI) * acos(P_miss_y / P_miss_magnitude) : 0;
            double phi_z = (180. / PI) * atan2(P_miss_z, P_miss_x);

            outfile << tot_fKE << "\",\"" << tot_hadronic_energy << "\",\"" << p_tot << "\",\"" << p_miss << "\",\"" << MissE << "\",\"" << P_miss_x << "\",\"" << P_miss_y << "\",\"" << P_miss_z << "\",\"" << topology << "\"\n";
        }

        // NumuInclusiveNpNpi/NueInclusiveNpNpi

        if (found && (!is_Final_lepton_PDG_int && str_Final_lepton_PDG == "Inclusive") && Min_Nu_energy < fKE_nu && fKE_nu < Max_Nu_energy &&
            ((is_num_prot_int && is_num_pion_int && n_prot == int_num_prot && n_pi == int_num_pion) ||
             (!is_num_prot_int && str_num_prot == "N" && is_num_pion_int && n_pi == int_num_pion) ||
             (is_num_prot_int && n_prot == int_num_prot && !is_num_pion_int && str_num_pion == "N") ||
             (!is_num_prot_int && str_num_prot == "N" && !is_num_pion_int && str_num_pion == "N")))
        {

            // //std::cout << "I am here yay for i  "<<i<< std::endl;

            for (int j = 0; j < tStdHepN; j++)
            {
                // //std::cout <<" i issss "<<i<< " and j isssss " << j << " and status is " << tStdHepStatus[j] <<" and pdg " << tStdHepPdg[j]<< std::endl;

                if (tStdHepStatus[j] == 0 && (tStdHepPdg[j] == -16 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16))
                {

                    auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);

                    nu_energy = fKE;
                    nu_px = tStdHepP4[4 * j];
                    nu_py = tStdHepP4[4 * j + 1];
                    nu_pz = tStdHepP4[4 * j + 2];

                    double baseline = calc_baseline(tStdHepP4, fAbsoluteParticleMomentum, j);
                    double phi = phi_nu(tStdHepP4, fAbsoluteParticleMomentum, j);

                    Init_Nu_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                    Init_Nu_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    Init_Nu_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    Init_Nu_Phi->Fill(phi);
                    Oscillogram->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum, fAbsoluteParticleMomentum);

                    // outfile << "\"" << i << "\",";
                    double theta_z_nu = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    double phi_z_nu = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                    outfile << "\"" << i << "\"," << std::setprecision(6) << "\"" << tStdHepPdg[j] << "\",\"" << fAbsoluteParticleMomentum << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << phi << "\",\"" << theta_z_nu << "\",\"" << phi_z_nu << "\",\"" << baseline << "\",\"";
                    // Output the results
                    // //std::cout << "i AM HEre for i:"<<i<<"and j:"<< j << std::endl;

                    // //std::cout << "Absolute Particle Momentum: " << fAbsoluteParticleMomentum << std::endl;
                    // // //std::cout << "Invariant Mass: " << fInvMass << std::endl;
                    // //std::cout << "Kinetic Energy: " << fKE << std::endl;
                }

                if (tStdHepStatus[j] == 1)
                {

                    auto [fAbsoluteParticleMomentum, fInvMass, fKE] = kinematics(tStdHepP4, j);
                    // Charged current cases
                    if (((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE >= muon_ke ||
                         (tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE >= electron_ke ||
                         (tStdHepPdg[j] == 15 || tStdHepPdg[j] == -15)) &&
                        j <= 5)
                    {

                        double theta_z_lep = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        double phi_z_lep = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                        outfile << std::setprecision(6) << tStdHepPdg[j] << "\",\"" << fInvMass << "\",\"" << fKE << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << theta_z_lep << "\",\"" << phi_z_lep << "\",\"";

                        lep_energy = fKE;
                        lep_px = tStdHepP4[4 * j];
                        lep_py = tStdHepP4[4 * j + 1];
                        lep_pz = tStdHepP4[4 * j + 2];

                        tot_fKE += fKE;
                        tot_fpx += tStdHepP4[4 * j];
                        tot_fpy += tStdHepP4[4 * j + 1];
                        tot_fpz += tStdHepP4[4 * j + 2];

                        Fin_CC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                        Fin_CC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        Fin_CC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    }

                    else if (((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE < muon_ke ||
                              (tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE < electron_ke) &&
                             j <= 5)
                    {

                        outfile << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"";
                    }

                    // Neutral current

                    else if ((tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -16) && j <= 5)
                    {
                        auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);
                        double theta_z_lep = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        double phi_z_lep = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                        outfile << std::setprecision(6) << tStdHepPdg[j] << "\",\"" << 0 << "\",\"" << fKE << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << theta_z_lep << "\",\"" << phi_z_lep << "\",\"";

                        Fin_NC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                        Fin_NC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        Fin_NC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    }

                    else if (tStdHepPdg[j] == 2212 || tStdHepPdg[j] == 211 || tStdHepPdg[j] == -211 || tStdHepPdg[j] == 111 || tStdHepPdg[j] == 321 || tStdHepPdg[j] == -321 || tStdHepPdg[j] == 130 || tStdHepPdg[j] == 310 || tStdHepPdg[j] == 22 || tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11 || tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13)
                    {
                        finalparticles_info(tStdHepP4, j, tStdHepPdg, pdgs, masses,
                                            energies, pxs, pys, pzs, costheta_arr, theta_arr, tot_fKE, tot_hadronic_energy, tot_fpx, tot_fpy, tot_fpz, dictionary);
                    }
                }
            }

            // //do_resize(pdgs,masses,energies,pxs,pys,pzs,costheta_arr,theta_arr);

            Fin_Prot_Mult->Fill(n_prot);
            n_prot = 0;
            Fin_PiPlus_Mult->Fill(n_piplus);
            n_piplus = 0;
            Fin_PiMinus_Mult->Fill(n_piminus);
            n_piminus = 0;
            Fin_PiZero_Mult->Fill(n_pizero);
            n_pizero = 0;

            if (pdgs.empty())
            {
                pdgs.push_back(0);
            }

            if (!pdgs.empty())
            {
                for (int j = 0; j < pdgs.size(); j++)
                {
                    if (j < pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << ",";
                    }
                    if (j == pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << "\"";
                    }
                }
            }
            outfile << ",\"";

            // Check if each array is empty, and if so, replace it with a single 0
            if (masses.empty())
                masses.push_back(0);
            if (energies.empty())
                energies.push_back(0);
            if (pxs.empty())
                pxs.push_back(0);
            if (pys.empty())
                pys.push_back(0);
            if (pzs.empty())
                pzs.push_back(0);
            if (costheta_arr.empty())
                costheta_arr.push_back(0);
            if (theta_arr.empty())
                theta_arr.push_back(0);

            // Now write each array to the file
            writeVectorToFile(outfile, masses);
            writeVectorToFile(outfile, energies);
            writeVectorToFile(outfile, pxs);
            writeVectorToFile(outfile, pys);
            writeVectorToFile(outfile, pzs);
            writeVectorToFile(outfile, costheta_arr);
            writeVectorToFile(outfile, theta_arr);

            double p_tot = sqrt(pow(tot_fpx, 2) + pow(tot_fpy, 2) + pow(tot_fpz, 2));
            double p_miss = tot_fKE - p_tot;
            double MissE = nu_energy - tot_fKE;
            double P_miss_x = nu_px - tot_fpx;
            double P_miss_y = nu_py - tot_fpy;
            double P_miss_z = nu_pz - tot_fpz;

            double P_miss_magnitude = sqrt(pow(P_miss_x, 2) + pow(P_miss_y, 2) + pow(P_miss_z, 2));
            double theta_z = (P_miss_magnitude > 0) ? (180. / PI) * acos(P_miss_y / P_miss_magnitude) : 0;
            double phi_z = (180. / PI) * atan2(P_miss_z, P_miss_x);

            outfile << tot_fKE << "\",\"" << tot_hadronic_energy << "\",\"" << p_tot << "\",\"" << p_miss << "\",\"" << MissE << "\",\"" << P_miss_x << "\",\"" << P_miss_y << "\",\"" << P_miss_z << "\",\"" << topology << "\"\n";
        }

        // NumuCCNpNpi/NueCCNpNpi
        if (found && (!is_Final_lepton_PDG_int && str_Final_lepton_PDG == "CC") && (lepton_pdg_instore == 11 || lepton_pdg_instore == -11 || lepton_pdg_instore == 13 || lepton_pdg_instore == -13 || lepton_pdg_instore == 15 || lepton_pdg_instore == -15) && Min_Nu_energy < fKE_nu && fKE_nu < Max_Nu_energy &&
            ((is_num_prot_int && is_num_pion_int && n_prot == int_num_prot && n_pi == int_num_pion) ||
             (!is_num_prot_int && str_num_prot == "N" && is_num_pion_int && n_pi == int_num_pion) ||
             (is_num_prot_int && n_prot == int_num_prot && !is_num_pion_int && str_num_pion == "N") ||
             (!is_num_prot_int && str_num_prot == "N" && !is_num_pion_int && str_num_pion == "N")))
        {

            // std::cout << "I am here yay for i  "<<i<< std::endl;

            for (int j = 0; j < tStdHepN; j++)
            {
                // //std::cout <<" i issss "<<i<< " and j isssss " << j << " and status is " << tStdHepStatus[j] <<" and pdg " << tStdHepPdg[j]<< std::endl;

                if (tStdHepStatus[j] == 0 && (tStdHepPdg[j] == -16 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16))
                {

                    auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);

                    nu_energy = fKE;
                    nu_px = tStdHepP4[4 * j];
                    nu_py = tStdHepP4[4 * j + 1];
                    nu_pz = tStdHepP4[4 * j + 2];

                    double baseline = calc_baseline(tStdHepP4, fAbsoluteParticleMomentum, j);
                    double phi = phi_nu(tStdHepP4, fAbsoluteParticleMomentum, j);

                    Init_Nu_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                    Init_Nu_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    Init_Nu_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    Init_Nu_Phi->Fill(phi);
                    Oscillogram->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum, fAbsoluteParticleMomentum);

                    // outfile << "\"" << i << "\",";
                    double theta_z_nu = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    double phi_z_nu = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                    outfile << "\"" << i << "\"," << std::setprecision(6) << "\"" << tStdHepPdg[j] << "\",\"" << fAbsoluteParticleMomentum << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << phi << "\",\"" << theta_z_nu << "\",\"" << phi_z_nu << "\",\"" << baseline << "\",\"";
                    // Output the results
                    // //std::cout << "i AM HEre for i:"<<i<<"and j:"<< j << std::endl;

                    // //std::cout << "Absolute Particle Momentum: " << fAbsoluteParticleMomentum << std::endl;
                    // // //std::cout << "Invariant Mass: " << fInvMass << std::endl;
                    // //std::cout << "Kinetic Energy: " << fKE << std::endl;
                }

                if (tStdHepStatus[j] == 1)
                {
                    auto [fAbsoluteParticleMomentum, fInvMass, fKE] = kinematics(tStdHepP4, j);
                    // Charged current cases
                    if (((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE >= muon_ke ||
                         (tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE >= electron_ke ||
                         (tStdHepPdg[j] == 15 || tStdHepPdg[j] == -15)) &&
                        j <= 5)
                    {

                        double theta_z_lep = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        double phi_z_lep = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                        outfile << std::setprecision(6) << tStdHepPdg[j] << "\",\"" << fInvMass << "\",\"" << fKE << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << theta_z_lep << "\",\"" << phi_z_lep << "\",\"";

                        lep_energy = fKE;
                        lep_px = tStdHepP4[4 * j];
                        lep_py = tStdHepP4[4 * j + 1];
                        lep_pz = tStdHepP4[4 * j + 2];

                        tot_fKE += fKE;
                        tot_fpx += tStdHepP4[4 * j];
                        tot_fpy += tStdHepP4[4 * j + 1];
                        tot_fpz += tStdHepP4[4 * j + 2];

                        Fin_CC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                        Fin_CC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        Fin_CC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    }

                    else if (((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE < muon_ke ||
                              (tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE < electron_ke) &&
                             j <= 5)
                    {

                        outfile << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"";
                    }

                    else if (tStdHepPdg[j] == 2212 || tStdHepPdg[j] == 211 || tStdHepPdg[j] == -211 || tStdHepPdg[j] == 111 || tStdHepPdg[j] == 321 || tStdHepPdg[j] == -321 || tStdHepPdg[j] == 130 || tStdHepPdg[j] == 310 || tStdHepPdg[j] == 22 || tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11 || tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13)
                    {
                        finalparticles_info(tStdHepP4, j, tStdHepPdg, pdgs, masses,
                                            energies, pxs, pys, pzs, costheta_arr, theta_arr, tot_fKE, tot_hadronic_energy, tot_fpx, tot_fpy, tot_fpz, dictionary);
                    }
                }
            }

            // ////do_resize(pdgs,masses,energies,pxs,pys,pzs,costheta_arr,theta_arr);

            if (pdgs.empty())
            {
                pdgs.push_back(0);
            }

            if (!pdgs.empty())
            {
                for (int j = 0; j < pdgs.size(); j++)
                {
                    if (j < pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << ",";
                    }
                    if (j == pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << "\"";
                    }
                }
            }
            outfile << ",\"";

            // Check if each array is empty, and if so, replace it with a single 0
            if (masses.empty())
                masses.push_back(0);
            if (energies.empty())
                energies.push_back(0);
            if (pxs.empty())
                pxs.push_back(0);
            if (pys.empty())
                pys.push_back(0);
            if (pzs.empty())
                pzs.push_back(0);
            if (costheta_arr.empty())
                costheta_arr.push_back(0);
            if (theta_arr.empty())
                theta_arr.push_back(0);

            // Now write each array to the file
            writeVectorToFile(outfile, masses);
            writeVectorToFile(outfile, energies);
            writeVectorToFile(outfile, pxs);
            writeVectorToFile(outfile, pys);
            writeVectorToFile(outfile, pzs);
            writeVectorToFile(outfile, costheta_arr);
            writeVectorToFile(outfile, theta_arr);

            double p_tot = sqrt(pow(tot_fpx, 2) + pow(tot_fpy, 2) + pow(tot_fpz, 2));
            double p_miss = tot_fKE - p_tot;
            double MissE = nu_energy - tot_fKE;
            double P_miss_x = nu_px - tot_fpx;
            double P_miss_y = nu_py - tot_fpy;
            double P_miss_z = nu_pz - tot_fpz;

            double P_miss_magnitude = sqrt(pow(P_miss_x, 2) + pow(P_miss_y, 2) + pow(P_miss_z, 2));
            double theta_z = (P_miss_magnitude > 0) ? (180. / PI) * acos(P_miss_y / P_miss_magnitude) : 0;
            double phi_z = (180. / PI) * atan2(P_miss_z, P_miss_x);

            outfile << tot_fKE << "\",\"" << tot_hadronic_energy << "\",\"" << p_tot << "\",\"" << p_miss << "\",\"" << MissE << "\",\"" << P_miss_x << "\",\"" << P_miss_y << "\",\"" << P_miss_z << "\",\"" << topology << "\"\n";
        }

        // NumuNCNpNpi/NueNCNpNpi
        if (found && (!is_Final_lepton_PDG_int && str_Final_lepton_PDG == "NC") && (lepton_pdg_instore == 12 || lepton_pdg_instore == -12 || lepton_pdg_instore == 14 || lepton_pdg_instore == -14 || lepton_pdg_instore == 16 || lepton_pdg_instore == -16) && Min_Nu_energy < fKE_nu && fKE_nu < Max_Nu_energy &&
            ((is_num_prot_int && is_num_pion_int && n_prot == int_num_prot && n_pi == int_num_pion) ||
             (!is_num_prot_int && str_num_prot == "N" && is_num_pion_int && n_pi == int_num_pion) ||
             (is_num_prot_int && n_prot == int_num_prot && !is_num_pion_int && str_num_pion == "N") ||
             (!is_num_prot_int && str_num_prot == "N" && !is_num_pion_int && str_num_pion == "N")))
        {

            // std::cout << "I am here yay for i  "<<i<< std::endl;

            for (int j = 0; j < tStdHepN; j++)
            {
                // //std::cout <<" i issss "<<i<< " and j isssss " << j << " and status is " << tStdHepStatus[j] <<" and pdg " << tStdHepPdg[j]<< std::endl;

                if (tStdHepStatus[j] == 0 && (tStdHepPdg[j] == -16 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16))
                {

                    auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);

                    nu_energy = fKE;
                    nu_px = tStdHepP4[4 * j];
                    nu_py = tStdHepP4[4 * j + 1];
                    nu_pz = tStdHepP4[4 * j + 2];

                    double baseline = calc_baseline(tStdHepP4, fAbsoluteParticleMomentum, j);
                    double phi = phi_nu(tStdHepP4, fAbsoluteParticleMomentum, j);

                    Init_Nu_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                    Init_Nu_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    Init_Nu_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    Init_Nu_Phi->Fill(phi);
                    Oscillogram->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum, fAbsoluteParticleMomentum);

                    // outfile << "\"" << i << "\",";
                    double theta_z_nu = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    double phi_z_nu = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                    outfile << "\"" << i << "\"," << std::setprecision(6) << "\"" << tStdHepPdg[j] << "\",\"" << fAbsoluteParticleMomentum << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << phi << "\",\"" << theta_z_nu << "\",\"" << phi_z_nu << "\",\"" << baseline << "\",\"";
                }

                if (tStdHepStatus[j] == 1)
                {

                    if ((tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -16) && j <= 5)
                    {
                        auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);
                        double theta_z_lep = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        double phi_z_lep = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                        outfile << std::setprecision(6) << tStdHepPdg[j] << "\",\"" << 0 << "\",\"" << fKE << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << theta_z_lep << "\",\"" << phi_z_lep << "\",\"";

                        Fin_NC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                        Fin_NC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        Fin_NC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    }

                    else if (tStdHepPdg[j] == 2212 || tStdHepPdg[j] == 211 || tStdHepPdg[j] == -211 || tStdHepPdg[j] == 111 || tStdHepPdg[j] == 321 || tStdHepPdg[j] == -321 || tStdHepPdg[j] == 130 || tStdHepPdg[j] == 310 || tStdHepPdg[j] == 22 || tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11 || tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13)
                    {
                        finalparticles_info(tStdHepP4, j, tStdHepPdg, pdgs, masses,
                                            energies, pxs, pys, pzs, costheta_arr, theta_arr, tot_fKE, tot_hadronic_energy, tot_fpx, tot_fpy, tot_fpz, dictionary);
                    }
                }
            }

            // do_resize(pdgs,masses,energies,pxs,pys,pzs,costheta_arr,theta_arr);

            Fin_Prot_Mult->Fill(n_prot);
            n_prot = 0;
            Fin_PiPlus_Mult->Fill(n_piplus);
            n_piplus = 0;
            Fin_PiMinus_Mult->Fill(n_piminus);
            n_piminus = 0;
            Fin_PiZero_Mult->Fill(n_pizero);
            n_pizero = 0;

            if (pdgs.empty())
            {
                pdgs.push_back(0);
            }

            if (!pdgs.empty())
            {
                for (int j = 0; j < pdgs.size(); j++)
                {
                    if (j < pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << ",";
                    }
                    if (j == pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << "\"";
                    }
                }
            }
            outfile << ",\"";

            // Check if each array is empty, and if so, replace it with a single 0
            if (masses.empty())
                masses.push_back(0);
            if (energies.empty())
                energies.push_back(0);
            if (pxs.empty())
                pxs.push_back(0);
            if (pys.empty())
                pys.push_back(0);
            if (pzs.empty())
                pzs.push_back(0);
            if (costheta_arr.empty())
                costheta_arr.push_back(0);
            if (theta_arr.empty())
                theta_arr.push_back(0);

            // Now write each array to the file
            writeVectorToFile(outfile, masses);
            writeVectorToFile(outfile, energies);
            writeVectorToFile(outfile, pxs);
            writeVectorToFile(outfile, pys);
            writeVectorToFile(outfile, pzs);
            writeVectorToFile(outfile, costheta_arr);
            writeVectorToFile(outfile, theta_arr);

            double p_tot = sqrt(pow(tot_fpx, 2) + pow(tot_fpy, 2) + pow(tot_fpz, 2));
            double p_miss = tot_fKE - p_tot;
            double MissE = nu_energy - tot_fKE;
            double P_miss_x = nu_px - tot_fpx;
            double P_miss_y = nu_py - tot_fpy;
            double P_miss_z = nu_pz - tot_fpz;

            double P_miss_magnitude = sqrt(pow(P_miss_x, 2) + pow(P_miss_y, 2) + pow(P_miss_z, 2));
            double theta_z = (P_miss_magnitude > 0) ? (180. / PI) * acos(P_miss_y / P_miss_magnitude) : 0;
            double phi_z = (180. / PI) * atan2(P_miss_z, P_miss_x);

            outfile << tot_fKE << "\",\"" << tot_hadronic_energy << "\",\"" << p_tot << "\",\"" << p_miss << "\",\"" << MissE << "\",\"" << P_miss_x << "\",\"" << P_miss_y << "\",\"" << P_miss_z << "\",\"" << topology << "\"\n";
        }

        // AnyNuCCNpNpi
        if (string_found && (!is_Final_lepton_PDG_int && str_Final_lepton_PDG == "CC") && (lepton_pdg_instore == 11 || lepton_pdg_instore == -11 || lepton_pdg_instore == 13 || lepton_pdg_instore == -13 || lepton_pdg_instore == 15 || lepton_pdg_instore == -15) && Min_Nu_energy < fKE_nu && fKE_nu < Max_Nu_energy &&
            ((is_num_prot_int && is_num_pion_int && n_prot == int_num_prot && n_pi == int_num_pion) ||
             (!is_num_prot_int && str_num_prot == "N" && is_num_pion_int && n_pi == int_num_pion) ||
             (is_num_prot_int && n_prot == int_num_prot && !is_num_pion_int && str_num_pion == "N") ||
             (!is_num_prot_int && str_num_prot == "N" && !is_num_pion_int && str_num_pion == "N")))
        {

            // std::cout << "I am here yay for i  "<<i<< std::endl;

            for (int j = 0; j < tStdHepN; j++)
            {

                if (tStdHepStatus[j] == 0 && (tStdHepPdg[j] == -16 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16))
                {

                    auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);

                    nu_energy = fKE;
                    nu_px = tStdHepP4[4 * j];
                    nu_py = tStdHepP4[4 * j + 1];
                    nu_pz = tStdHepP4[4 * j + 2];

                    double baseline = calc_baseline(tStdHepP4, fAbsoluteParticleMomentum, j);
                    double phi = phi_nu(tStdHepP4, fAbsoluteParticleMomentum, j);

                    Init_Nu_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                    Init_Nu_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    Init_Nu_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    Init_Nu_Phi->Fill(phi);
                    Oscillogram->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum, fAbsoluteParticleMomentum);

                    // outfile << "\"" << i << "\",";
                    double theta_z_nu = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    double phi_z_nu = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                    outfile << "\"" << i << "\"," << std::setprecision(6) << "\"" << tStdHepPdg[j] << "\",\"" << fAbsoluteParticleMomentum << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << phi << "\",\"" << theta_z_nu << "\",\"" << phi_z_nu << "\",\"" << baseline << "\",\"";

                    // Output the results
                    // //std::cout << "i AM HEre for i:"<<i<<"and j:"<< j << std::endl;

                    // //std::cout << "Absolute Particle Momentum: " << fAbsoluteParticleMomentum << std::endl;
                    // // //std::cout << "Invariant Mass: " << fInvMass << std::endl;
                    // //std::cout << "Kinetic Energy: " << fKE << std::endl;
                }

                if (tStdHepStatus[j] == 1)
                {
                    auto [fAbsoluteParticleMomentum, fInvMass, fKE] = kinematics(tStdHepP4, j);

                    // Charged current cases
                    if (((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE >= muon_ke ||
                         (tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE >= electron_ke ||
                         (tStdHepPdg[j] == 15 || tStdHepPdg[j] == -15)) &&
                        j <= 5)
                    {

                        double theta_z_lep = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        double phi_z_lep = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                        outfile << std::setprecision(6) << tStdHepPdg[j] << "\",\"" << fInvMass << "\",\"" << fKE << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << theta_z_lep << "\",\"" << phi_z_lep << "\",\"";

                        lep_energy = fKE;
                        lep_px = tStdHepP4[4 * j];
                        lep_py = tStdHepP4[4 * j + 1];
                        lep_pz = tStdHepP4[4 * j + 2];

                        tot_fKE += fKE;
                        tot_fpx += tStdHepP4[4 * j];
                        tot_fpy += tStdHepP4[4 * j + 1];
                        tot_fpz += tStdHepP4[4 * j + 2];

                        Fin_CC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                        Fin_CC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        Fin_CC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    }

                    else if (((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE < muon_ke ||
                              (tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE < electron_ke) &&
                             j <= 5)
                    {

                        outfile << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"";
                    }

                    else if (tStdHepPdg[j] == 2212 || tStdHepPdg[j] == 211 || tStdHepPdg[j] == -211 || tStdHepPdg[j] == 111 || tStdHepPdg[j] == 321 || tStdHepPdg[j] == -321 || tStdHepPdg[j] == 130 || tStdHepPdg[j] == 310 || tStdHepPdg[j] == 22 || tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11 || tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13)
                    {
                        finalparticles_info(tStdHepP4, j, tStdHepPdg, pdgs, masses,
                                            energies, pxs, pys, pzs, costheta_arr, theta_arr, tot_fKE, tot_hadronic_energy, tot_fpx, tot_fpy, tot_fpz, dictionary);
                    }
                }
            }

            // //do_resize(pdgs,masses,energies,pxs,pys,pzs,costheta_arr,theta_arr);

            Fin_Prot_Mult->Fill(n_prot);
            n_prot = 0;
            Fin_PiPlus_Mult->Fill(n_piplus);
            n_piplus = 0;
            Fin_PiMinus_Mult->Fill(n_piminus);
            n_piminus = 0;
            Fin_PiZero_Mult->Fill(n_pizero);
            n_pizero = 0;

            if (pdgs.empty())
            {
                pdgs.push_back(0);
            }

            if (!pdgs.empty())
            {
                for (int j = 0; j < pdgs.size(); j++)
                {
                    if (j < pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << ",";
                    }
                    if (j == pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << "\"";
                    }
                }
            }
            outfile << ",\"";

            // Check if each array is empty, and if so, replace it with a single 0
            if (masses.empty())
                masses.push_back(0);
            if (energies.empty())
                energies.push_back(0);
            if (pxs.empty())
                pxs.push_back(0);
            if (pys.empty())
                pys.push_back(0);
            if (pzs.empty())
                pzs.push_back(0);
            if (costheta_arr.empty())
                costheta_arr.push_back(0);
            if (theta_arr.empty())
                theta_arr.push_back(0);

            // Now write each array to the file
            writeVectorToFile(outfile, masses);
            writeVectorToFile(outfile, energies);
            writeVectorToFile(outfile, pxs);
            writeVectorToFile(outfile, pys);
            writeVectorToFile(outfile, pzs);
            writeVectorToFile(outfile, costheta_arr);
            writeVectorToFile(outfile, theta_arr);

            double p_tot = sqrt(pow(tot_fpx, 2) + pow(tot_fpy, 2) + pow(tot_fpz, 2));
            double p_miss = tot_fKE - p_tot;
            double MissE = nu_energy - tot_fKE;
            double P_miss_x = nu_px - tot_fpx;
            double P_miss_y = nu_py - tot_fpy;
            double P_miss_z = nu_pz - tot_fpz;

            double P_miss_magnitude = sqrt(pow(P_miss_x, 2) + pow(P_miss_y, 2) + pow(P_miss_z, 2));
            double theta_z = (P_miss_magnitude > 0) ? (180. / PI) * acos(P_miss_y / P_miss_magnitude) : 0;
            double phi_z = (180. / PI) * atan2(P_miss_z, P_miss_x);

            outfile << tot_fKE << "\",\"" << tot_hadronic_energy << "\",\"" << p_tot << "\",\"" << p_miss << "\",\"" << MissE << "\",\"" << P_miss_x << "\",\"" << P_miss_y << "\",\"" << P_miss_z << "\",\"" << topology << "\"\n";
        }

        // AnyNuNCNpNpi

        if (string_found && (!is_Final_lepton_PDG_int && str_Final_lepton_PDG == "NC") && (lepton_pdg_instore == 12 || lepton_pdg_instore == -12 || lepton_pdg_instore == 14 || lepton_pdg_instore == -14 || lepton_pdg_instore == 16 || lepton_pdg_instore == -16) && Min_Nu_energy < fKE_nu && fKE_nu < Max_Nu_energy &&
            ((is_num_prot_int && is_num_pion_int && n_prot == int_num_prot && n_pi == int_num_pion) ||
             (!is_num_prot_int && str_num_prot == "N" && is_num_pion_int && n_pi == int_num_pion) ||
             (is_num_prot_int && n_prot == int_num_prot && !is_num_pion_int && str_num_pion == "N") ||
             (!is_num_prot_int && str_num_prot == "N" && !is_num_pion_int && str_num_pion == "N")))
        {

            // std::cout << "I am here yay for i  "<<i<< std::endl;

            for (int j = 0; j < tStdHepN; j++)
            {
                // //std::cout <<" i issss "<<i<< " and j isssss " << j << " and status is " << tStdHepStatus[j] <<" and pdg " << tStdHepPdg[j]<< std::endl;

                if (tStdHepStatus[j] == 0 && (tStdHepPdg[j] == -16 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16))
                {

                    auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);

                    nu_energy = fKE;
                    nu_px = tStdHepP4[4 * j];
                    nu_py = tStdHepP4[4 * j + 1];
                    nu_pz = tStdHepP4[4 * j + 2];

                    double baseline = calc_baseline(tStdHepP4, fAbsoluteParticleMomentum, j);
                    double phi = phi_nu(tStdHepP4, fAbsoluteParticleMomentum, j);

                    Init_Nu_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                    Init_Nu_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    Init_Nu_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    Init_Nu_Phi->Fill(phi);
                    Oscillogram->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum, fAbsoluteParticleMomentum);

                    // outfile << "\"" << i << "\",";
                    double theta_z_nu = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    double phi_z_nu = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                    outfile << "\"" << i << "\"," << std::setprecision(6) << "\"" << tStdHepPdg[j] << "\",\"" << fAbsoluteParticleMomentum << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << phi << "\",\"" << theta_z_nu << "\",\"" << phi_z_nu << "\",\"" << baseline << "\",\"";
                }

                if (tStdHepStatus[j] == 1)
                {

                    if ((tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -16) && j <= 5)
                    {
                        auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);
                        double theta_z_lep = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        double phi_z_lep = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                        outfile << std::setprecision(6) << tStdHepPdg[j] << "\",\"" << 0 << "\",\"" << fKE << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << theta_z_lep << "\",\"" << phi_z_lep << "\",\"";

                        Fin_NC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                        Fin_NC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        Fin_NC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    }

                    else if (tStdHepPdg[j] == 2212 || tStdHepPdg[j] == 211 || tStdHepPdg[j] == -211 || tStdHepPdg[j] == 111 || tStdHepPdg[j] == 321 || tStdHepPdg[j] == -321 || tStdHepPdg[j] == 130 || tStdHepPdg[j] == 310 || tStdHepPdg[j] == 22 || tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11 || tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13)
                    {
                        finalparticles_info(tStdHepP4, j, tStdHepPdg, pdgs, masses,
                                            energies, pxs, pys, pzs, costheta_arr, theta_arr, tot_fKE, tot_hadronic_energy, tot_fpx, tot_fpy, tot_fpz, dictionary);
                    }
                }
            }

            // //do_resize(pdgs,masses,energies,pxs,pys,pzs,costheta_arr,theta_arr);

            Fin_Prot_Mult->Fill(n_prot);
            n_prot = 0;
            Fin_PiPlus_Mult->Fill(n_piplus);
            n_piplus = 0;
            Fin_PiMinus_Mult->Fill(n_piminus);
            n_piminus = 0;
            Fin_PiZero_Mult->Fill(n_pizero);
            n_pizero = 0;

            if (pdgs.empty())
            {
                pdgs.push_back(0);
            }

            if (!pdgs.empty())
            {
                for (int j = 0; j < pdgs.size(); j++)
                {
                    if (j < pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << ",";
                    }
                    if (j == pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << "\"";
                    }
                }
            }
            outfile << ",\"";

            // Check if each array is empty, and if so, replace it with a single 0
            if (masses.empty())
                masses.push_back(0);
            if (energies.empty())
                energies.push_back(0);
            if (pxs.empty())
                pxs.push_back(0);
            if (pys.empty())
                pys.push_back(0);
            if (pzs.empty())
                pzs.push_back(0);
            if (costheta_arr.empty())
                costheta_arr.push_back(0);
            if (theta_arr.empty())
                theta_arr.push_back(0);

            // Now write each array to the file
            writeVectorToFile(outfile, masses);
            writeVectorToFile(outfile, energies);
            writeVectorToFile(outfile, pxs);
            writeVectorToFile(outfile, pys);
            writeVectorToFile(outfile, pzs);
            writeVectorToFile(outfile, costheta_arr);
            writeVectorToFile(outfile, theta_arr);

            double p_tot = sqrt(pow(tot_fpx, 2) + pow(tot_fpy, 2) + pow(tot_fpz, 2));
            double p_miss = tot_fKE - p_tot;
            double MissE = nu_energy - tot_fKE;
            double P_miss_x = nu_px - tot_fpx;
            double P_miss_y = nu_py - tot_fpy;
            double P_miss_z = nu_pz - tot_fpz;

            double P_miss_magnitude = sqrt(pow(P_miss_x, 2) + pow(P_miss_y, 2) + pow(P_miss_z, 2));
            double theta_z = (P_miss_magnitude > 0) ? (180. / PI) * acos(P_miss_y / P_miss_magnitude) : 0;
            double phi_z = (180. / PI) * atan2(P_miss_z, P_miss_x);

            outfile << tot_fKE << "\",\"" << tot_hadronic_energy << "\",\"" << p_tot << "\",\"" << p_miss << "\",\"" << MissE << "\",\"" << P_miss_x << "\",\"" << P_miss_y << "\",\"" << P_miss_z << "\",\"" << topology << "\"\n";
        }

        // AnyNuInclusiveNpNpi(Basically All)
        if (string_found && (!is_Final_lepton_PDG_int && str_Final_lepton_PDG == "Inclusive") && Min_Nu_energy < fKE_nu && fKE_nu < Max_Nu_energy &&
            ((is_num_prot_int && is_num_pion_int && n_prot == int_num_prot && n_pi == int_num_pion) ||
             (!is_num_prot_int && str_num_prot == "N" && is_num_pion_int && n_pi == int_num_pion) ||
             (is_num_prot_int && n_prot == int_num_prot && !is_num_pion_int && str_num_pion == "N") ||
             (!is_num_prot_int && str_num_prot == "N" && !is_num_pion_int && str_num_pion == "N")))
        {

            // std::cout << "I am here yay for i  "<<i<< std::endl;

            for (int j = 0; j < tStdHepN; j++)
            {
                // //std::cout <<" i issss "<<i<< " and j isssss " << j << " and status is " << tStdHepStatus[j] <<" and pdg " << tStdHepPdg[j]<< std::endl;

                if (tStdHepStatus[j] == 0 && (tStdHepPdg[j] == -16 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16))
                {
                    auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);

                    nu_energy = fKE;
                    nu_px = tStdHepP4[4 * j];
                    nu_py = tStdHepP4[4 * j + 1];
                    nu_pz = tStdHepP4[4 * j + 2];

                    double baseline = calc_baseline(tStdHepP4, fAbsoluteParticleMomentum, j);
                    double phi = phi_nu(tStdHepP4, fAbsoluteParticleMomentum, j);

                    Init_Nu_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                    Init_Nu_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    Init_Nu_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    Init_Nu_Phi->Fill(phi);
                    Oscillogram->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum, fAbsoluteParticleMomentum);

                    // outfile << "\"" << i << "\",";
                    double theta_z_nu = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                    double phi_z_nu = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                    outfile << "\"" << i << "\"," << std::setprecision(6) << "\"" << tStdHepPdg[j] << "\",\"" << fAbsoluteParticleMomentum << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << phi << "\",\"" << theta_z_nu << "\",\"" << phi_z_nu << "\",\"" << baseline << "\",\"";

                    // outfile << "\"" << i << "\"," << std::setprecision(6)  << "\"" << tStdHepPdg[j] << "\",\"" << fAbsoluteParticleMomentum << "\",\"" << tStdHepP4[4*j] << "\",\"" << tStdHepP4[4*j+1] << "\",\"" << tStdHepP4[4*j+2] << "\",\"" << tStdHepP4[4*j+1]/fAbsoluteParticleMomentum << "\",\"" << (180./PI)*acos(tStdHepP4[4*j+1]/fAbsoluteParticleMomentum) << "\",\"" ;
                }

                if (tStdHepStatus[j] == 1)
                {
                    auto [fAbsoluteParticleMomentum, fInvMass, fKE] = kinematics(tStdHepP4, j);
                    // Charged current cases
                    if (((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE >= muon_ke ||
                         (tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE >= electron_ke ||
                         (tStdHepPdg[j] == 15 || tStdHepPdg[j] == -15)) &&
                        j <= 5)
                    {

                        double theta_z_lep = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        double phi_z_lep = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                        outfile << std::setprecision(6) << tStdHepPdg[j] << "\",\"" << fInvMass << "\",\"" << fKE << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << theta_z_lep << "\",\"" << phi_z_lep << "\",\"";

                        lep_energy = fKE;
                        lep_px = tStdHepP4[4 * j];
                        lep_py = tStdHepP4[4 * j + 1];
                        lep_pz = tStdHepP4[4 * j + 2];

                        tot_fKE += fKE;
                        tot_fpx += tStdHepP4[4 * j];
                        tot_fpy += tStdHepP4[4 * j + 1];
                        tot_fpz += tStdHepP4[4 * j + 2];

                        Fin_CC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                        Fin_CC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        Fin_CC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    }
                    else if (((tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13) && fKE < muon_ke ||
                              (tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11) && fKE < electron_ke) &&
                             j <= 5)
                    {

                        outfile << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"" << 0 << "\",\"";
                    }

                    // Neutral current cases
                    else if ((tStdHepPdg[j] == 12 || tStdHepPdg[j] == 14 || tStdHepPdg[j] == 16 || tStdHepPdg[j] == -12 || tStdHepPdg[j] == -14 || tStdHepPdg[j] == -16) && j <= 5)
                    {
                        auto [fAbsoluteParticleMomentum, fKE] = kinematics_massless(tStdHepP4, j);
                        double theta_z_lep = (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        double phi_z_lep = atan2(tStdHepP4[4 * j], tStdHepP4[4 * j + 2]) * (180. / PI);

                        outfile << std::setprecision(6) << tStdHepPdg[j] << "\",\"" << 0 << "\",\"" << fKE << "\",\"" << tStdHepP4[4 * j] << "\",\"" << tStdHepP4[4 * j + 1] << "\",\"" << tStdHepP4[4 * j + 2] << "\",\"" << tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum << "\",\"" << (180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum) << "\",\"" << theta_z_lep << "\",\"" << phi_z_lep << "\",\"";

                        Fin_NC_Lept_Mom->Fill(1000. * fAbsoluteParticleMomentum);
                        Fin_NC_Lept_CosTheta->Fill(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum);
                        Fin_NC_Lept_Theta->Fill((180. / PI) * acos(tStdHepP4[4 * j + 1] / fAbsoluteParticleMomentum));
                    }

                    else if (tStdHepPdg[j] == 2212 || tStdHepPdg[j] == 211 || tStdHepPdg[j] == -211 || tStdHepPdg[j] == 111 || tStdHepPdg[j] == 321 || tStdHepPdg[j] == -321 || tStdHepPdg[j] == 130 || tStdHepPdg[j] == 310 || tStdHepPdg[j] == 22 || tStdHepPdg[j] == 11 || tStdHepPdg[j] == -11 || tStdHepPdg[j] == 13 || tStdHepPdg[j] == -13)
                    {
                        finalparticles_info(tStdHepP4, j, tStdHepPdg, pdgs, masses,
                                            energies, pxs, pys, pzs, costheta_arr, theta_arr, tot_fKE, tot_hadronic_energy, tot_fpx, tot_fpy, tot_fpz, dictionary);
                    }
                }
            }

            // //do_resize(pdgs,masses,energies,pxs,pys,pzs,costheta_arr,theta_arr);

            Fin_Prot_Mult->Fill(n_prot);
            n_prot = 0;
            Fin_PiPlus_Mult->Fill(n_piplus);
            n_piplus = 0;
            Fin_PiMinus_Mult->Fill(n_piminus);
            n_piminus = 0;
            Fin_PiZero_Mult->Fill(n_pizero);
            n_pizero = 0;

            if (pdgs.empty())
            {
                pdgs.push_back(0);
            }

            if (!pdgs.empty())
            {
                for (int j = 0; j < pdgs.size(); j++)
                {
                    if (j < pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << ",";
                    }
                    if (j == pdgs.size() - 1)
                    {
                        outfile << setprecision(6) << pdgs[j] << "\"";
                    }
                }
            }
            outfile << ",\"";

            // Check if each array is empty, and if so, replace it with a single 0
            if (masses.empty())
                masses.push_back(0);
            if (energies.empty())
                energies.push_back(0);
            if (pxs.empty())
                pxs.push_back(0);
            if (pys.empty())
                pys.push_back(0);
            if (pzs.empty())
                pzs.push_back(0);
            if (costheta_arr.empty())
                costheta_arr.push_back(0);
            if (theta_arr.empty())
                theta_arr.push_back(0);

            // Now write each array to the file
            writeVectorToFile(outfile, masses);
            writeVectorToFile(outfile, energies);
            writeVectorToFile(outfile, pxs);
            writeVectorToFile(outfile, pys);
            writeVectorToFile(outfile, pzs);
            writeVectorToFile(outfile, costheta_arr);
            writeVectorToFile(outfile, theta_arr);

            double p_tot = sqrt(pow(tot_fpx, 2) + pow(tot_fpy, 2) + pow(tot_fpz, 2));
            double p_miss = tot_fKE - p_tot;
            double MissE = nu_energy - tot_fKE;
            double P_miss_x = nu_px - tot_fpx;
            double P_miss_y = nu_py - tot_fpy;
            double P_miss_z = nu_pz - tot_fpz;

            double P_miss_magnitude = sqrt(pow(P_miss_x, 2) + pow(P_miss_y, 2) + pow(P_miss_z, 2));
            double theta_z = (P_miss_magnitude > 0) ? (180. / PI) * acos(P_miss_y / P_miss_magnitude) : 0;
            double phi_z = (180. / PI) * atan2(P_miss_z, P_miss_x);

            outfile << tot_fKE << "\",\"" << tot_hadronic_energy << "\",\"" << p_tot << "\",\"" << p_miss << "\",\"" << MissE << "\",\"" << P_miss_x << "\",\"" << P_miss_y << "\",\"" << P_miss_z << "\",\"" << topology << "\"\n";
        }

    } // End of event loop

    // /////////////////////////////////////////////////FOR RESIZING VECTOR/////////////////
    // std::cout <<"final max num: "<<max_num<< endl;

    // Create a map to store strings as keys and integers as values
    // std::map<std::string, int> maxprong_dict;

    // // Insert key-value pairs into the map
    // maxprong_dict[outfile_name] = max_num;

    // // Open a file in write mode
    // std::ofstream file("maxprong_dict.txt", std::ios_base::app);

    // // Write key-value pairs to the file
    // for (const auto& pair :maxprong_dict) {
    //     file << pair.first << " " << pair.second << std::endl;
    // }

    // Write out histograms to file
    Init_Nu_Mom->Write();
    Init_Nu_CosTheta->Write();
    Init_Nu_Theta->Write();
    Init_Nu_Phi->Write();
    Oscillogram->Write();
    Fin_CC_Lept_Mom->Write();
    Fin_CC_Lept_Theta->Write();
    Fin_CC_Lept_CosTheta->Write();
    Fin_NC_Lept_Mom->Write();
    Fin_NC_Lept_Theta->Write();
    Fin_NC_Lept_CosTheta->Write();
    Fin_Prot_Mom->Write();
    Fin_PiPlus_Mom->Write();
    Fin_PiMinus_Mom->Write();
    Fin_PiZero_Mom->Write();
    Fin_Gamma_Mom->Write();

    // std::cout << "Initial Neutrino Momentum, AR23" << endl;
    // Init_Nu_Mom->Print("all");
    Init_Nu_Mom->Draw("hist");
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->SetLogy(1);
    c1->Print((directory + "/" + last_name + "Init_Nu_Mom.png").c_str());
    c1->Print((directory + "/" + last_name + "Init_Nu_Mom.root").c_str());
    c1->Clear();

    // Now for the Nu angle histograms
    Init_Nu_CosTheta->Draw("hist");
    c1->SetLogy(0);
    c1->Print((directory + "/" + last_name + "Init_Nu_CosTheta.png").c_str());
    c1->Print((directory + "/" + last_name + "Init_Nu_CosTheta.root").c_str());
    c1->Clear();

    // And now for the initial neutrino oscillogram
    Init_Nu_Theta->Draw("hist");
    c1->SetLogy(0);
    c1->Print((directory + "/" + last_name + "Init_Nu_Theta.png").c_str());
    c1->Print((directory + "/" + last_name + "Init_Nu_Theta.root").c_str());
    c1->Clear();

    Init_Nu_Phi->Draw("hist");
    c1->SetLogy(0);
    c1->Print((directory + "/" + last_name + "Init_Nu_Phi.png").c_str());
    c1->Print((directory + "/" + last_name + "Init_Nu_Phi.root").c_str());
    c1->Clear();

    Oscillogram->Draw("colz");
    c1->SetLogy(1);
    c1->SetLogz(1);
    c1->Print((directory + "/" + last_name + "Oscillogram.png").c_str());
    c1->Print((directory + "/" + last_name + "Oscillogram.root").c_str());
    c1->Clear();

    Fin_CC_Lept_Mom->Draw("hist");
    c1->SetLogy(1);
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_Mom.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_Mom.root").c_str());
    c1->Clear();

    Fin_CC_Lept_Theta->Draw("hist");
    c1->SetLogy(0);
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_Theta.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_Theta.root").c_str());
    c1->Clear();

    Fin_CC_Lept_CosTheta->Draw("hist");
    c1->SetLogy(0);
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_CosTheta.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_CosTheta.root").c_str());
    c1->Clear();

    Fin_NC_Lept_Mom->Draw("hist");
    c1->SetLogy(1);
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_Mom.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_Mom.root").c_str());
    c1->Clear();

    Fin_NC_Lept_Theta->Draw("hist");
    c1->SetLogy(0);
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_Theta.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_Theta.root").c_str());
    c1->Clear();

    Fin_NC_Lept_CosTheta->Draw("hist");
    c1->SetLogy(0);
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_CosTheta.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_CC_Lept_CosTheta.root").c_str());
    c1->Clear();

    Fin_Prot_Mom->Draw("hist");
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->SetLogy(1);
    c1->Print((directory + "/" + last_name + "Fin_Prot_Mom.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_Prot_Mom.root").c_str());
    // c1->Print((directory + "/" + last_name + "Fin_Prot_Mom_low.png").c_str());
    // c1->Print((directory + "/" + last_name + "Fin_Prot_Mom_low.root").c_str());
    c1->Clear();

    Fin_Prot_Mult->Draw("hist");
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->SetLogy(0);
    c1->Print((directory + "/" + last_name + "Fin_Prot_Mult.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_Prot_Mult.root").c_str());
    c1->Clear();

    Fin_PiPlus_Mom->Draw("hist");
    c1->SetLogy(1);
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->Print((directory + "/" + last_name + "Fin_PiPlus_Mom.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_PiPlus_Mom.root").c_str());
    // c1->Print((directory + "/" + last_name + "Fin_PiPlus_Mom_low.png").c_str());
    // c1->Print((directory + "/" + last_name + "Fin_PiPlus_Mom_low.root").c_str());
    c1->Clear();

    Fin_PiPlus_Mult->Draw("hist");
    c1->SetLogy(0);
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->Print((directory + "/" + last_name + "Fin_PiPlus_Mult.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_PiPlus_Mult.root").c_str());
    c1->Clear();

    Fin_PiMinus_Mom->Draw("hist");
    c1->SetLogy(1);
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->Print((directory + "/" + last_name + "Fin_PiMinus_Mom.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_PiMinus_Mom.root").c_str());
    // c1->Print((directory + "/" + last_name + "Fin_PiMinus_Mom_low.png").c_str());
    // c1->Print((directory + "/" + last_name + "Fin_PiMinus_Mom_low.root").c_str());
    c1->Clear();

    Fin_PiMinus_Mult->Draw("hist");
    c1->SetLogy(0);
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->Print((directory + "/" + last_name + "Fin_PiMinus_Mult.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_PiMinus_Mult.root").c_str());
    c1->Clear();

    // pi0 histograms
    Fin_PiZero_Mom->Draw("hist");
    c1->SetLogy(1);
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->Print((directory + "/" + last_name + "Fin_PiZero_Mom.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_PiZero_Mom.root").c_str());
    // c1->Print((directory + "/" + last_name + "Fin_PiZero_Mom_low.png").c_str());
    // c1->Print((directory + "/" + last_name + "Fin_PiZero_Mom_Low.root").c_str());
    c1->Clear();

    Fin_PiZero_Mult->Draw("hist");
    c1->SetLogy(0);
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->Print((directory + "/" + last_name + "Fin_PiZero_Mult.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_PiZero_Mult.root").c_str());
    c1->Clear();

    Fin_Gamma_Mom->Draw("hist");
    c1->BuildLegend(0.5, 0.3, 0.9, 0.7);
    c1->SetLogy(1);
    c1->Print((directory + "/" + last_name + "Fin_Gamma_Mom.png").c_str());
    c1->Print((directory + "/" + last_name + "Fin_Gamma_Mom.root").c_str());
    c1->Clear();

    treefile->Write();
    c1->Close();
}

// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
//         return 1; // Exit with error code 1 if incorrect usage
//     }

//     std::string input_file = argv[1];
//     ScalarLepton(input_file);

//     return 0; // Exit successfully
// }

void run_test(const char *input_file)
{
    std::string file(input_file);
    ScalarLept_wNC(file);
}

// double fAbsoluteParticleMomentum, fInvMass, fKE;
// kinematics(tStdHepP4, j, fAbsoluteParticleMomentum, fInvMass, fKE);
