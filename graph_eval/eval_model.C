// eval_model.C
// Updated: integrates 2D contour + ellipse + fractionInsideEllipse (writes ellipse_fraction.txt)
// Preserves original functionality.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <set>
#include <numeric>
#include <limits>

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TStyle.h"
#include "TDirectory.h"
#include "TLine.h"
#include "TEllipse.h"
#include "TROOT.h"
#include "TObjArray.h"
#include "TList.h"
#include "TGraph.h"
#include "TPad.h"
#include "TSystem.h"

/// === Configurable parameters ===
const int NUM_BINS = 50;
const double XMIN_DEFAULT = -1.0;
const double XMAX_DEFAULT = -1.0;

const double PI = 3.141592653589793;
const double R = 6371000;
const double h = 15000;

/// === Math helpers ===
double mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / v.size();
}

double rms(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return sqrt(s / v.size());
}

double stddev(const std::vector<double>& v) {
    double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return sqrt(s / v.size());
}

/// === Physics Calculations ===
double calcTheta(double x, double y, double z) {
    double p = sqrt(x*x + y*y + z*z);
    return (p == 0) ? 0 : (180.0 / PI) * acos(y / p);
}

double calcCosTheta(double x, double y, double z) {
    double p = sqrt(x*x + y*y + z*z);
    return (p == 0) ? 0 : y / p;
}

double calc_baseline(const double Nu_Mom_X, const double Nu_Mom_Y, const double Nu_Mom_Z) {
    double Nu_Mom_Tot = sqrt(Nu_Mom_X * Nu_Mom_X + Nu_Mom_Y * Nu_Mom_Y + Nu_Mom_Z * Nu_Mom_Z);
    if (Nu_Mom_Tot == 0) return 0.0;
    double theta_z = acos(Nu_Mom_Y / Nu_Mom_Tot);
    double eta = PI - theta_z;
    return (sqrt(pow(R + h, 2) - pow(R * sin(eta), 2)) + (R * cos(eta))) / 1000.0;
}

/// === Contour / ellipse helpers (adapted from plot_2d_hist_contour.C) ===
std::vector<double> calcLevels(TH2D* h, const std::vector<double>& probs) {
    std::vector<double> vals;
    for (int ix=1; ix<=h->GetNbinsX(); ++ix)
        for (int iy=1; iy<=h->GetNbinsY(); ++iy)
            vals.push_back(h->GetBinContent(ix,iy));
    std::sort(vals.begin(), vals.end(), std::greater<double>());
    double total = std::accumulate(vals.begin(), vals.end(), 0.0);

    std::vector<double> cumsum(vals.size());
    std::partial_sum(vals.begin(), vals.end(), cumsum.begin());

    std::vector<double> levels;
    for (double p : probs) {
        double target = p * total;
        auto it = std::lower_bound(cumsum.begin(), cumsum.end(), target);
        if (it != cumsum.end())
            levels.push_back(vals[std::distance(cumsum.begin(), it)]);
    }
    std::sort(levels.begin(), levels.end());
    return levels;
}

/// fraction of histogram population inside an ellipse centered at (x0,y0)
double fractionInsideEllipse(const TH2* h,
                             double x0, double y0,
                             double a,  double b,
                             double theta = 0.0)
{
    double total = 0.0, inside = 0.0;
    for (int ix=1; ix<=h->GetNbinsX(); ++ix) {
        for (int iy=1; iy<=h->GetNbinsY(); ++iy) {
            double x = h->GetXaxis()->GetBinCenter(ix);
            double y = h->GetYaxis()->GetBinCenter(iy);
            double w = h->GetBinContent(ix,iy);
            total += w;

            // shift to ellipse center
            double dx = x - x0;
            double dy = y - y0;
            // rotate by -theta (theta in radians)
            double xr =  dx*std::cos(theta) + dy*std::sin(theta);
            double yr = -dx*std::sin(theta) + dy*std::cos(theta);

            if ( (xr*xr)/(a*a) + (yr*yr)/(b*b) <= 1.0 )
                inside += w;
        }
    }
    return (total>0) ? inside/total : 0.0;
}

double polygonArea(const TGraph* g) {
    if (!g || g->GetN() < 3) return 0.0;
    double area = 0.0;
    const int n = g->GetN();
    const double* x = g->GetX();
    const double* y = g->GetY();
    for (int i=0, j=n-1; i<n; j=i++) {
        area += (x[j]*y[i] - x[i]*y[j]);
    }
    return std::fabs(area) * 0.5;
}

// Helper to append/update ellipse_fraction.csv
void updateEllipseCSV(const std::string& modelName,
                      double ell_a,
                      double ell_b,
                      double frac)
{
    const std::string filename = "ellipse_fraction.csv";

    // Build column name for this ellipse (no commas to keep CSV simple)
    std::ostringstream ellNameOSS;
    ellNameOSS << "Fraction inside ellipse (center 0 0; a=" << (int)ell_a
               << "; b=" << (int)ell_b << ")";
    std::string ellColName = ellNameOSS.str();

    // Data structures for CSV in memory
    std::vector<std::string> colNames;
    std::vector< std::vector<std::string> > rows;

    bool fileExists = !gSystem->AccessPathName(filename.c_str());

    if (fileExists) {
        std::ifstream fin(filename.c_str());
        if (!fin.is_open()) {
            std::cerr << "Could not open " << filename << " for reading.\n";
            fileExists = false;
        } else {
            // Read header line
            std::string line;
            if (std::getline(fin, line)) {
                std::stringstream ss(line);
                std::string cell;
                while (std::getline(ss, cell, ',')) {
                    colNames.push_back(cell);
                }
            }

            // Read data rows
            std::string line2;
            while (std::getline(fin, line2)) {
                std::stringstream ss(line2);
                std::string cell;
                std::vector<std::string> row;
                while (std::getline(ss, cell, ',')) {
                    row.push_back(cell);
                }
                rows.push_back(row);
            }
            fin.close();
        }
    }

    // If file didn't exist or couldn't be read, start fresh with header
    if (!fileExists) {
        colNames.clear();
        rows.clear();
        colNames.push_back("model_name");
        // ellipse column will be added below
    }

    // Helper to find or create a column
    auto getColIndex = [&](const std::string& name) -> int {
        for (size_t i = 0; i < colNames.size(); ++i) {
            if (colNames[i] == name) return static_cast<int>(i);
        }
        return -1;
    };

    // Ensure model_name column exists
    int modelCol = getColIndex("model_name");
    if (modelCol == -1) {
        colNames.push_back("model_name");
        modelCol = static_cast<int>(colNames.size()) - 1;
        // pad existing rows with empty cells
        for (auto& r : rows) {
            r.resize(colNames.size(), "");
        }
    }

    // Ensure ellipse-specific column exists
    int ellCol = getColIndex(ellColName);
    if (ellCol == -1) {
        colNames.push_back(ellColName);
        ellCol = static_cast<int>(colNames.size()) - 1;
        // pad existing rows with empty cells
        for (auto& r : rows) {
            r.resize(colNames.size(), "");
        }
    }

    // Create new row with all columns
    std::vector<std::string> newRow(colNames.size(), "");
    newRow[modelCol] = modelName;
    newRow[ellCol] = std::to_string(frac);

    rows.push_back(std::move(newRow));

    // Write everything back to ellipse_fraction.csv (overwrite)
    std::ofstream fout(filename.c_str());
    if (!fout.is_open()) {
        std::cerr << "Could not open " << filename << " for writing.\n";
        return;
    }

    // Header
    for (size_t i = 0; i < colNames.size(); ++i) {
        if (i) fout << ",";
        fout << colNames[i];
    }
    fout << "\n";

    // Rows
    for (const auto& r : rows) {
        for (size_t i = 0; i < colNames.size(); ++i) {
            if (i) fout << ",";
            if (i < r.size()) fout << r[i];
        }
        fout << "\n";
    }

    fout.close();
    std::cout << "Updated " << filename << " with model_name=" << modelName
              << " and " << ellColName << " = " << frac << std::endl;
}

/// === 2D resolution graphing helper ===
void graph_resolution_stat(
    const std::string& base_var,
    const std::vector<double>& x_data,
    const std::vector<double>& resolution,
    const std::string& stat2,
    int bins = NUM_BINS,
    double xmin = XMIN_DEFAULT,
    double xmax = XMAX_DEFAULT,
    TDirectory* outdir = nullptr
) {
    if (x_data.empty() || resolution.empty()) return;

    if (xmin == xmax) {
        xmin = *std::min_element(x_data.begin(), x_data.end());
        xmax = *std::max_element(x_data.begin(), x_data.end());
    }
    if (xmin == xmax) { // avoid zero width
        xmin -= 0.5; xmax += 0.5;
    }

    double bin_width = (xmax - xmin) / bins;
    std::vector<std::vector<double>> bin_y(bins);
    std::vector<double> bin_x(bins), bin_yval(bins), bin_yerr(bins), bin_xerr(bins);

    for (size_t i = 0; i < x_data.size(); ++i) {
        int bin = static_cast<int>( (x_data[i] - xmin) / bin_width );
        if (bin >= 0 && bin < bins)
            bin_y[bin].push_back(resolution[i]);
    }

    for (int i = 0; i < bins; ++i) {
        bin_x[i] = xmin + (i + 0.5) * bin_width;
        bin_xerr[i] = bin_width / 2;
        if (!bin_y[i].empty()) {
            bin_yval[i] = mean(bin_y[i]);
            bin_yerr[i] = (stat2 == "rms") ? rms(bin_y[i]) : stddev(bin_y[i]);
        } else {
            bin_yval[i] = 0;
            bin_yerr[i] = 0;
        }
    }

    // compute automatic y-range from mean ± error, then clamp to ±200% 
    double ymin =  std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < bins; ++i) {
        // Only consider bins that actually had entries
        if (!bin_y[i].empty()) {
            double ylow  = bin_yval[i] - bin_yerr[i];
            double yhigh = bin_yval[i] + bin_yerr[i];
            if (ylow  < ymin) ymin  = ylow;
            if (yhigh > ymax) ymax = yhigh;
        }
    }

    // Fallback if everything was empty for some reason
    if (!std::isfinite(ymin) || !std::isfinite(ymax)) {
        ymin = -1.0;
        ymax =  1.0;
    }

    // Clamp to ±200%
    const double LIM = 200.0;
    if (ymin < -LIM) ymin = -LIM;
    if (ymax >  LIM) ymax =  LIM;

    // Avoid zero-width range after clamping
    if (ymin == ymax) {
        ymin -= 1.0;
        ymax += 1.0;
    }

    std::string name = "gr_" + base_var + "_" + stat2;
    TCanvas* c = new TCanvas(name.c_str(), name.c_str(), 800, 600);
    gStyle->SetOptStat(0);
    TGraphErrors* gr = new TGraphErrors(bins, &bin_x[0], &bin_yval[0], &bin_xerr[0], &bin_yerr[0]);
    gr->SetMarkerStyle(20);
    gr->SetMarkerColor(kOrange + 7);
    gr->SetLineColor(kOrange + 7);
    gr->SetTitle((base_var + ": mean #pm " + stat2).c_str());
    gr->GetXaxis()->SetTitle(base_var.c_str());
    gr->GetYaxis()->SetTitle("Resolution (%)");
    gr->GetYaxis()->SetRangeUser(ymin, ymax);

    if (outdir) outdir->cd(); // ensure writing to correct directory
    gr->Write(name.c_str());
    c->Write((name + "_canvas").c_str());
    std::cout << "Writing graph to: " << gDirectory->GetPath() << std::endl;
}

// === Helper to clamp resolution axis limits to ±200% ==
std::pair<double,double> clamp_res_range(double xmin, double xmax) {
    constexpr double LIM = 200.0;
    xmin = std::max(xmin, -LIM);
    xmax = std::min(xmax,  LIM);

    // In case range is degenerate after clamping
    if (xmin == xmax) {
        xmin -= 1.0;
        xmax += 1.0;
    }
    return {xmin, xmax};
}

// === utilities for 2D truth-vs-reco plots ===
std::pair<double,double> finite_minmax(const std::vector<double>& v) {
    double vmin =  std::numeric_limits<double>::infinity();
    double vmax = -std::numeric_limits<double>::infinity();
    for (double x : v) {
        if (std::isnan(x) || std::isinf(x)) continue;
        if (x < vmin) vmin = x;
        if (x > vmax) vmax = x;
    }
    if (!std::isfinite(vmin) || !std::isfinite(vmax)) { vmin = -1.0; vmax = 1.0; }
    if (vmin == vmax) { vmin -= 0.5; vmax += 0.5; }
    return {vmin, vmax};
}

void plot_truth_vs_reco_2d(const std::string& base,
                            const std::vector<double>& vtrue,
                            const std::vector<double>& vreco,
                            TDirectory* outdir,
                            int nbins = 200)
{
    if (vtrue.empty() || vreco.empty()) return;

    // Determine a common range that covers both truth and reco (nice for the y=x line)
    auto [tmin, tmax] = finite_minmax(vtrue);
    auto [pmin, pmax] = finite_minmax(vreco);
    double xmin = std::min(tmin, pmin);
    double xmax = std::max(tmax, pmax);

    // Small padding
    double pad = 0.05 * std::max(1.0, std::fabs(xmax - xmin));
    xmin -= pad; xmax += pad;

    // Build histogram (x = truth, y = reco)
    std::string hname = "h2_" + base + "_reco_vs_true";
    std::string htitle = base + ": reconstructed vs true;true " + base + ";reconstructed " + base;
    TH2D* h2 = new TH2D(hname.c_str(), htitle.c_str(), nbins, xmin, xmax, nbins, xmin, xmax);
    h2->SetDirectory(0);

    size_t N = std::min(vtrue.size(), vreco.size());
    for (size_t i = 0; i < N; ++i) {
        double t = vtrue[i], p = vreco[i];
        if (std::isnan(t) || std::isnan(p) || std::isinf(t) || std::isinf(p)) continue;
        h2->Fill(t, p);
    }

    // Draw
    std::string cname = "c2_" + base + "_reco_vs_true";
    TCanvas* c = new TCanvas(cname.c_str(), cname.c_str(), 900, 800);
    gStyle->SetOptStat(0);
    h2->Draw("COLZ");

    // y = x line for reference
    TLine* diag = new TLine(xmin, xmin, xmax, xmax);
    diag->SetLineColor(kBlack);
    diag->SetLineStyle(2);
    diag->SetLineWidth(2);
    diag->Draw("SAME");

    // Persist to the desired directory
    if (outdir) outdir->cd();
    h2->Write(hname.c_str());
    c->Write((cname + "_canvas").c_str());
}

void drawLargestContourAtLevel(TH2D* h, double level, int lineColor, int lineWidth=3) {
    // Keep track of the current visible pad
    TVirtualPad* saved = gPad;

    // Make a throwaway clone with just this level
    std::unique_ptr<TH2D> hc((TH2D*)h->Clone(("__cont_tmp_"+std::to_string((long long)level)).c_str()));
    hc->SetDirectory(0);
    hc->SetContour(1);
    hc->SetContourLevel(0, level);

    // Build contours on a hidden canvas to avoid nuking the visible pad
    TCanvas junk("__cont_workpad","__cont_workpad",10,10);
    junk.cd();
    hc->Draw("CONT Z LIST");
    junk.Update();

    // Fetch contour lists
    TObjArray* conts = (TObjArray*) gROOT->GetListOfSpecials()->FindObject("contours");
    if (!conts || conts->GetEntries() == 0) return;
    TList* levelList = (TList*) conts->At(0); // one level
    if (!levelList || levelList->GetSize() == 0) { 
        gROOT->GetListOfSpecials()->Remove(conts);
        delete conts;
        return; 
    }

    // Pick largest-area loop
    TGraph* best = nullptr;
    double bestArea = -1.0;
    TIter it(levelList);
    while (TObject* o = it()) {
        TGraph* g = dynamic_cast<TGraph*>(o);
        if (!g) continue;
        double a = polygonArea(g);
        if (a > bestArea) { bestArea = a; best = g; }
    }

    // Draw chosen loop on the original visible pad
    if (best) {
        saved->cd();
        best->SetLineColor(lineColor);
        best->SetLineWidth(lineWidth);
        best->Draw("L SAME");
    }

    // Clean specials
    gROOT->GetListOfSpecials()->Remove(conts);
    delete conts;
}

/// === Main Function ===
void eval_model(const char* filename) {

     // Put ROOT into batch mode so canvases are not shown on screen
    Bool_t oldBatch = gROOT->IsBatch();
    gROOT->SetBatch(kTRUE);

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        gROOT->SetBatch(oldBatch); // restore batch state before returning
        return;
    }

    // === Parse header ===
    std::string line;
    if (!std::getline(infile, line)) {
        std::cerr << "Empty file or missing header: " << filename << std::endl;
        return;
    }
    std::stringstream header_ss(line);
    std::string token;
    std::vector<std::string> headers;
    std::map<std::string,int> colIndex;

    int col = 0;
    while (std::getline(header_ss, token, ',')) {
        // trim whitespace from token (minimal)
        size_t b = token.find_first_not_of(" \t\r\n");
        size_t e = token.find_last_not_of(" \t\r\n");
        std::string key = (b==std::string::npos) ? std::string() : token.substr(b, e-b+1);
        headers.push_back(key);
        colIndex[key] = col++;
    }

    // === Extract directory name from last column header ===
    // Example: "GENIEv3-0-6-Honda-Truth-hA-LFG_Numu_CC_Thresh_p1to1_eventnum_All_NpNpi_MSE_E_Px_Py_Pz_EID"
    std::string dirName = headers.empty()
        ? std::string("Output")
        : headers.back();

    // (optional) trim whitespace if needed
    {
        size_t b = dirName.find_first_not_of(" \t\r\n");
        size_t e = dirName.find_last_not_of(" \t\r\n");
        if (b == std::string::npos) dirName = "Output";
        else dirName = dirName.substr(b, e - b + 1);
    }

    // === Open output file (append if it exists) ===
    bool fileExists = !gSystem->AccessPathName("combined_output.root"); 
    // AccessPathName returns kTRUE if "inaccessible", so ! means "exists and accessible"

    TFile* outfile = nullptr;
    if (fileExists) {
        outfile = TFile::Open("combined_output.root", "UPDATE");
        std::cout << "Opened existing combined_output.root in UPDATE mode.\n";
    } else {
        outfile = TFile::Open("combined_output.root", "RECREATE");
        std::cout << "Created new combined_output.root.\n";
    }

    if (!outfile || outfile->IsZombie()) {
        std::cerr << "Error opening combined_output.root" << std::endl;
        return;
    }

    // === Get or create the directory for this run ===
    TDirectory* plotDir = outfile->GetDirectory(dirName.c_str());

    if (!plotDir) {
        // Directory does not exist → make it
        plotDir = outfile->mkdir(dirName.c_str());
        std::cout << "[INFO] Created new directory: " << dirName << std::endl;
    } else {
        // Directory already exists → WARNING
        std::cout << "=========================================================\n";
        std::cout << "[WARNING] Directory already exists in combined_output.root:\n";
        std::cout << "          " << dirName << "\n";
        std::cout << "          New plots will be appended or overwrite cycles.\n";
        std::cout << "=========================================================\n";
    }

    plotDir->cd();  // Always cd to the directory
    std::cout << "Directory after plotDir->cd(): " << gDirectory->GetPath() << std::endl;


    // === Prepare containers ===
    std::map<std::string, std::vector<double>> data;

    std::vector<double> Cos_Theta_nu_pred, Theta_nu_pred;
    std::vector<double> Cos_Theta_nu_true, Theta_nu_true;
    std::vector<double> pred_Mass_squared, true_Mass_squared;
    std::vector<double> pred_baseline, true_baseline;

    // === Read data rows ===
    static int line_num = 1;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string field;
        while (std::getline(ss, field, ',')) {
            if (field.empty()) {
                row.push_back(NAN);  // preserve column count
            } else {
                try {
                    row.push_back(std::stod(field));
                } catch (...) {
                    row.push_back(NAN);
                }
            }
        }

        if (row.size() != headers.size()) {
            std::cerr << "Skipping line " << line_num << ": column mismatch ("
                      << row.size() << " vs " << headers.size() << ")\n";
            ++line_num;
            continue;
        }
        ++line_num;

        for (size_t i = 0; i < headers.size(); ++i)
            data[headers[i]].push_back(row[i]);

        // fill derived vectors when momentum columns are present
        auto has = [&](const std::string& key) {
            return colIndex.find(key) != colIndex.end();
        };

        if (has("pred_Nu_Mom_X") && has("pred_Nu_Mom_Y") && has("pred_Nu_Mom_Z")) {
            double px = row[colIndex["pred_Nu_Mom_X"]];
            double py = row[colIndex["pred_Nu_Mom_Y"]];
            double pz = row[colIndex["pred_Nu_Mom_Z"]];
            Cos_Theta_nu_pred.push_back(calcCosTheta(px, py, pz));
            Theta_nu_pred.push_back(calcTheta(px, py, pz));
            pred_baseline.push_back(calc_baseline(px, py, pz));
        }

        if (has("true_Nu_Mom_X") && has("true_Nu_Mom_Y") && has("true_Nu_Mom_Z")) {
            double tx = row[colIndex["true_Nu_Mom_X"]];
            double ty = row[colIndex["true_Nu_Mom_Y"]];
            double tz = row[colIndex["true_Nu_Mom_Z"]];
            Cos_Theta_nu_true.push_back(calcCosTheta(tx, ty, tz));
            Theta_nu_true.push_back(calcTheta(tx, ty, tz));
            true_baseline.push_back(calc_baseline(tx, ty, tz));
        }

        if (has("true_Nu_Energy") && has("true_Nu_Mom_X") && has("true_Nu_Mom_Y") && has("true_Nu_Mom_Z")) {
            double E = row[colIndex["true_Nu_Energy"]];
            double px = row[colIndex["true_Nu_Mom_X"]];
            double py = row[colIndex["true_Nu_Mom_Y"]];
            double pz = row[colIndex["true_Nu_Mom_Z"]];
            true_Mass_squared.push_back(E*E - (px*px + py*py + pz*pz));
        }

        if (has("pred_Nu_Energy") && has("pred_Nu_Mom_X") && has("pred_Nu_Mom_Y") && has("pred_Nu_Mom_Z")) {
            double E = row[colIndex["pred_Nu_Energy"]];
            double px = row[colIndex["pred_Nu_Mom_X"]];
            double py = row[colIndex["pred_Nu_Mom_Y"]];
            double pz = row[colIndex["pred_Nu_Mom_Z"]];
            pred_Mass_squared.push_back(E*E - (px*px + py*py + pz*pz));
        }
    }
    infile.close();

    // === Create 1D histograms ===
auto make_hist = [&](const std::string& name, const std::vector<double>& d, double xmin, double xmax) {
    if (d.empty()) return;
    TH1D* h = new TH1D(name.c_str(), name.c_str(), 500, xmin, xmax);
    h->SetDirectory(0);  // prevent ROOT from auto-managing this hist
    for (double val : d) {
        if (!std::isnan(val)) h->Fill(val);
    }
    if (plotDir) plotDir->cd();
    h->Write();
    std::cout << "Writing hist to: " << gDirectory->GetPath() << " / " << name << std::endl;
};

    auto find_range = [](const std::vector<double>& d) {
        auto [minIt, maxIt] = std::minmax_element(d.begin(), d.end());
        double margin = 0.05 * std::max(std::abs(*minIt), std::abs(*maxIt));
        return std::make_pair(*minIt - margin, *maxIt + margin);
    };

    for (const auto& kv : data) {
        if (kv.second.empty()) continue;
        auto r = find_range(kv.second);
        make_hist(kv.first, kv.second, r.first, r.second);
    }

    make_hist("Cos_Theta_nu_pred", Cos_Theta_nu_pred, -1, 1);
    make_hist("Theta_nu_pred", Theta_nu_pred, 0, 180);
    make_hist("Cos_Theta_nu_true", Cos_Theta_nu_true, -1, 1);
    make_hist("Theta_nu_true", Theta_nu_true, 0, 180);
    make_hist("true_Mass_squared", true_Mass_squared, -2, 2);
    make_hist("pred_Mass_squared", pred_Mass_squared, -2, 2);
    make_hist("pred_baseline", pred_baseline, 0, 20000);
    make_hist("true_baseline", true_baseline, 0, 20000);

    // === Create resolution graphs ===
    for (const auto& kv : data) {
        if (kv.first.find("true_") != 0) continue;
        std::string base = kv.first.substr(5);
        std::string predName = "pred_" + base;
        if (data.find(predName) == data.end()) continue;

        std::vector<double> true_vals = data.at(kv.first);
        std::vector<double> pred_vals = data.at(predName);
        std::vector<double> res;

        for (size_t i = 0; i < true_vals.size(); ++i) {
            if (!std::isnan(true_vals[i]) && !std::isnan(pred_vals[i])) {
                if (true_vals[i] != 0)
                    res.push_back(100.0 * (pred_vals[i] - true_vals[i]) / true_vals[i]);
                else
                    std::cout << "Divide by zero at " << kv.first << " index " << i << std::endl;
            }
        }

        // 1D histogram of percent resolution for this variable ---
        if (!res.empty()) {
            auto r = find_range(res);
            auto cr = clamp_res_range(r.first, r.second);

            std::string hname  = "h1_res_" + base;
            std::string htitle = base + " percent resolution;Percent resolution (%);Counts";

            TH1D* hres = new TH1D(hname.c_str(), htitle.c_str(),
                                200, cr.first, cr.second); // 200 bins, auto range
            hres->SetDirectory(0); // keep hist independent of current gDirectory

            for (double v : res) {
                if (!std::isnan(v)) hres->Fill(v);
            }

            if (plotDir) plotDir->cd();
            hres->Write();
            std::cout << "Wrote 1D resolution hist: " << hname
                    << " to " << gDirectory->GetPath() << std::endl;
        }

        // 2D resolution plots with rms or std error bars
        graph_resolution_stat(base, true_vals, res, "rms", NUM_BINS, XMIN_DEFAULT, XMAX_DEFAULT, plotDir);
        graph_resolution_stat(base, true_vals, res, "std", NUM_BINS, XMIN_DEFAULT, XMAX_DEFAULT, plotDir);

        std::cout << "Writing graph to: " << gDirectory->GetPath() << std::endl;
    }


        // === For every true_*/pred_* pair, make 2D histogram of reco vs true ===
    {
        // iterate over all keys that start with "true_" and look for a sibling "pred_"
        int made2D = 0;
        for (const auto& kv : data) {
            const std::string& k = kv.first;
            if (k.rfind("true_", 0) != 0) continue; // not a true_* key

            std::string base = k.substr(5); // drop "true_"
            std::string predk = "pred_" + base;
            auto itp = data.find(predk);
            if (itp == data.end()) continue; // no matching pred_*

            const auto& vtrue = kv.second;
            const auto& vreco = itp->second;

            // If lengths differ, we safely use min size inside the helper
            plot_truth_vs_reco_2d(base, vtrue, vreco, plotDir, /*nbins=*/200);
            ++made2D;
        }
        std::cout << "Created " << made2D << " truth-vs-reco 2D histograms." << std::endl;
    }

    // === build 2D histogram for Energy% resolution vs theta diff ===
    // We'll compute pairs by iterating rows where both true/pred energy and true/pred momentum exist.
    std::vector<double> energy_res_percent;
    std::vector<double> theta_diff_signed;

    bool has_trueE = data.find("true_Nu_Energy") != data.end();
    bool has_predE = data.find("pred_Nu_Energy") != data.end();
    bool has_trueMom = (data.find("true_Nu_Mom_X") != data.end()
                        && data.find("true_Nu_Mom_Y") != data.end()
                        && data.find("true_Nu_Mom_Z") != data.end());
    bool has_predMom = (data.find("pred_Nu_Mom_X") != data.end()
                        && data.find("pred_Nu_Mom_Y") != data.end()
                        && data.find("pred_Nu_Mom_Z") != data.end());

    if (has_trueE && has_predE && has_trueMom && has_predMom) {
        size_t N = data["true_Nu_Energy"].size();
        // Sanity: make sure sizes match for columns used
        size_t NpredE = data["pred_Nu_Energy"].size();
        size_t NtrueMom = data["true_Nu_Mom_X"].size();
        size_t NpredMom = data["pred_Nu_Mom_X"].size();
        size_t Nmin = std::min({N, NpredE, NtrueMom, NpredMom});

        for (size_t i = 0; i < Nmin; ++i) {
            double Etrue = data["true_Nu_Energy"][i];
            double Epred = data["pred_Nu_Energy"][i];
            double tx = data["true_Nu_Mom_X"][i];
            double ty = data["true_Nu_Mom_Y"][i];
            double tz = data["true_Nu_Mom_Z"][i];
            double px = data["pred_Nu_Mom_X"][i];
            double py = data["pred_Nu_Mom_Y"][i];
            double pz = data["pred_Nu_Mom_Z"][i];

            if (std::isnan(Etrue) || std::isnan(Epred)) continue;
            if (Etrue == 0) continue; // avoid divide by zero

            double eres = 100.0 * (Epred - Etrue) / Etrue;
            double thet_true = calcTheta(tx, ty, tz);
            double thet_pred = calcTheta(px, py, pz);
            double tdiff = thet_pred - thet_true;

            if (!std::isnan(eres) && !std::isnan(tdiff)) {
                energy_res_percent.push_back(eres);
                theta_diff_signed.push_back(tdiff);
            }
        }
    } else {
        std::cout << "Skipping 2D energy/theta plot: missing columns (need true_Nu_Energy, pred_Nu_Energy, true_Nu_Mom_*, pred_Nu_Mom_*)." << std::endl;
    }

    // If we have entries for 2D, make the 2D histogram, draw contours and ellipse, compute fraction
    if (!energy_res_percent.empty() && !theta_diff_signed.empty()) {
        TCanvas* c2 = new TCanvas("energy_theta_2d", "Energy% vs |Delta Theta|", 900, 800);
        gStyle->SetOptStat(0);

        // Choose histogram ranges sensibly or derive from data
        double xmin = *std::min_element(energy_res_percent.begin(), energy_res_percent.end());
        double xmax = *std::max_element(energy_res_percent.begin(), energy_res_percent.end());

        auto cr = clamp_res_range(xmin, xmax);
        xmin = cr.first;
        xmax = cr.second;
        
        double ymin = -180.0; 
        double ymax =  180.0;

        // Expand slightly for nicer plotting
        double xpad = 0.05 * std::max(1.0, std::fabs(xmax - xmin));
        xmin -= xpad; 
        xmax += xpad;

        // Provide default wide x-range if values are too narrow
        if (xmin == xmax) { xmin -= 1; xmax += 1; }
        if (ymin == ymax) { ymin -= 1; ymax += 1; }

        TH2D* h2 = new TH2D("h2_energy_theta",
                            "Energy Resolution (%) vs #Delta#theta (deg);Energy Resolution (%) ;#Delta#theta (deg)",
                            200, xmin, xmax, 200, ymin, ymax);
        h2->SetDirectory(0);   // detach from gDirectory
        for (size_t i = 0; i < energy_res_percent.size(); ++i) {
            h2->Fill(energy_res_percent[i], theta_diff_signed[i]);
        }

        c2->cd();
        h2->Draw("COLZ");

        // Build a coarser/smoothed copy to stabilize contour topology
        TH2D* hcont = (TH2D*)h2->Clone("hcont_energy_theta");
        hcont->SetDirectory(0);
        hcont->Rebin2D(2, 2);   // mild coarsening
        hcont->Smooth(1);       // light smoothing (repeat if needed)

        // Compute HDR thresholds IN THIS ORDER: 95%, 90%, 68%
        std::vector<double> levels = calcLevels(hcont, {0.95, 0.90, 0.68});

        // Draw one closed loop (largest area) per level, color-coded
        if (levels.size() == 3) {
            drawLargestContourAtLevel(hcont, levels[0], kRed+1,    3); // 95% (outermost)
            drawLargestContourAtLevel(hcont, levels[1], kOrange+7, 3); // 90%
            drawLargestContourAtLevel(hcont, levels[2], kGreen+2,  3); // 68% (innermost)
        } else {
            for (size_t i = 0; i < levels.size(); ++i)
                drawLargestContourAtLevel(hcont, levels[i], kRed + (int)i, 3);
        }


        // Draw an ellipse centered at (0,0) with x semi-axis = 10 (%), y semi-axis = 30 (deg)
        // The requested ellipse: x-axis of 20% (±10%) and y-axis of 60 degrees (±30 degrees) =>
        // semi-axes a = 10, b = 30
        double ell_a = 10.0;
        double ell_b = 30.0;
        TEllipse* ell = new TEllipse(0.0, 0.0, ell_a, ell_b);
        ell->SetFillStyle(0);
        ell->SetLineColor(kBlue+2);
        ell->SetLineWidth(2);
        ell->Draw("SAME");

        // Compute fraction inside ellipse using the bin centers & weights
        double frac = fractionInsideEllipse(h2, 0.0, 0.0, ell_a, ell_b, 0.0 /*theta in radians*/);
        std::cout << "Fraction inside ellipse (a=" << ell_a << ", b=" << ell_b << "): " << frac << std::endl;

        // The model name is the last column header from result.csv (dirName above)
        std::string modelName = dirName;

        // Update ellipse_fraction.csv (create if needed, append a row otherwise)
        updateEllipseCSV(modelName, ell_a, ell_b, frac);

        // Write canvas and histogram to the selected directory
        if (plotDir) plotDir->cd();
        c2->Write("energy_theta_2d_canvas");
        h2->Write("h2_energy_theta");
    } else {
        std::cout << "No 2D entries available; energy/theta 2D histogram not created." << std::endl;
    }

    // Finish writing output file (preserve existing behavior)
    outfile->cd();
    outfile->Close();
    std::cout << "All outputs written to combined_output.root in directory: " 
              << dirName << std::endl;
    
    // Restore previous batch mode
    gROOT->SetBatch(oldBatch);

}
