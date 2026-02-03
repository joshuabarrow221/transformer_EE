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
#include "TTree.h"

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

double calcEnergyFromP(double px, double py, double pz) {
    if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) return NAN;
    return std::sqrt(px*px + py*py + pz*pz); // massless approximation
}

double calc_baseline(const double Nu_Mom_X, const double Nu_Mom_Y, const double Nu_Mom_Z) {
    double Nu_Mom_Tot = sqrt(Nu_Mom_X * Nu_Mom_X + Nu_Mom_Y * Nu_Mom_Y + Nu_Mom_Z * Nu_Mom_Z);
    if (Nu_Mom_Tot == 0) return 0.0;
    double theta_z = acos(Nu_Mom_Y / Nu_Mom_Tot);
    double eta = PI - theta_z;
    return (sqrt(pow(R + h, 2) - pow(R * sin(eta), 2)) + (R * cos(eta))) / 1000.0;
}

// Clamp helper
double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

// cos(theta) -> theta (deg), safe against slight |cos|>1
double thetaFromCos(double c) {
    if (std::isnan(c) || std::isinf(c)) return NAN;
    c = clamp(c, -1.0, 1.0);
    return (180.0 / PI) * std::acos(c);
}

bool isCosThetaVar(const std::string& base) {
    // base is the name WITHOUT true_/pred_ prefix (your "base" variable)
    // Add patterns you actually use in CSV headers.
    return (base.find("CosTheta")     != std::string::npos) ||
           (base.find("Cos_Theta")    != std::string::npos) ||
           (base.find("cosTheta")     != std::string::npos) ||
           (base.find("cos_theta")    != std::string::npos);
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
                      double frac,
                      double runtime_hours = -1.0)
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

    // Ensure runtime_hours column exists (W&B tracked runtime in hours)
    int rtCol = getColIndex("wandb_runtime_hours");
    if (rtCol == -1) {
        colNames.push_back("wandb_runtime_hours");
        rtCol = static_cast<int>(colNames.size()) - 1;
        for (auto& r : rows) {
            r.resize(colNames.size(), "");
        }
    }

    // Create new row with all columns
    std::vector<std::string> newRow(colNames.size(), "");
    newRow[modelCol] = modelName;
    newRow[ellCol] = std::to_string(frac);
    // Fill runtime only if provided and positive
    if (std::isfinite(runtime_hours) && runtime_hours > 0.0) {
        newRow[rtCol] = std::to_string(runtime_hours);
    }

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
    TDirectory* outdir = nullptr,
    bool is_percent = true
)
 {
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
    if (is_percent) {
        const double LIM = 200.0;
        if (ymin < -LIM) ymin = -LIM;
        if (ymax >  LIM) ymax =  LIM;
    }


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
    gr->GetYaxis()->SetTitle(is_percent ? "Resolution (%)" : "Residual (pred - true)");
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

// === Robust dynamic range helper using quantiles (reject outliers) ===
std::pair<double,double> quantile_range(const std::vector<double>& v,
                                        double qlo = 0.01, double qhi = 0.99,
                                        double padFrac = 0.10,
                                        double hardMin = -std::numeric_limits<double>::infinity(),
                                        double hardMax =  std::numeric_limits<double>::infinity(),
                                        double minWidth = 0.0)
{
    std::vector<double> w;
    w.reserve(v.size());
    for (double x : v) {
        if (std::isfinite(x)) w.push_back(x);
    }
    if (w.empty()) return {-1.0, 1.0};

    std::sort(w.begin(), w.end());

    auto pick = [&](double q) {
        if (q <= 0.0) return w.front();
        if (q >= 1.0) return w.back();
        double pos = q * (w.size() - 1);
        size_t i0 = (size_t)std::floor(pos);
        size_t i1 = std::min(i0 + 1, w.size() - 1);
        double t = pos - i0;
        return (1.0 - t) * w[i0] + t * w[i1];
    };

    double lo = pick(qlo);
    double hi = pick(qhi);

    // Clamp to hard limits if requested
    lo = std::max(lo, hardMin);
    hi = std::min(hi, hardMax);

    if (!std::isfinite(lo) || !std::isfinite(hi) || lo == hi) {
        lo = w.front();
        hi = w.back();
    }

    // Pad
    double width = hi - lo;
    if (!(width > 0.0)) width = 1.0;
    lo -= padFrac * width;
    hi += padFrac * width;

    // Enforce minimum width (useful for very narrow beam distributions)
    if (minWidth > 0.0 && (hi - lo) < minWidth) {
        double mid = 0.5 * (hi + lo);
        lo = mid - 0.5 * minWidth;
        hi = mid + 0.5 * minWidth;
    }

    // Final hard clamp
    lo = std::max(lo, hardMin);
    hi = std::min(hi, hardMax);

    if (lo == hi) { lo -= 1.0; hi += 1.0; }
    return {lo, hi};
}

// Optional: enforce symmetric range about zero (nice for Δθ)
std::pair<double,double> symmetric_about_zero(double lo, double hi, double minHalfWidth = 0.0)
{
    double a = std::max(std::fabs(lo), std::fabs(hi));
    if (a < minHalfWidth) a = minHalfWidth;
    return {-a, +a};
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

        // === Quantile-based common range (robust against outliers) ===
        double hardMin = -std::numeric_limits<double>::infinity();
        double hardMax =  std::numeric_limits<double>::infinity();
        double minWidth = 0.0;

        // Optional hard clamps for specific variables (based on "base" name)
        std::string b = base;
        std::transform(b.begin(), b.end(), b.begin(), ::tolower);

        // Identify momentum components robustly
        const bool isMom =
            (b.find("mom") != std::string::npos);

        // Identify energy robustly
        const bool isEnergy =
            (!isMom) && (
                b.find("energy") != std::string::npos);   // matches "nu_energy"

        // energy-like variables (GeV)
        if (isEnergy) {
            // Energy in GeV should not go negative
            hardMin = std::max(hardMin, 0.0);

            // Optional: enforce a minimum visible width so zoom doesn't collapse
            minWidth = std::max(minWidth, 0.2); // GeV, adjust as you like
        }

        // cos(theta)-like variables
        if (b.find("cos") != std::string::npos) {
            hardMin = -1.0;
            hardMax =  1.0;
            minWidth = 0.05;
        }

        // theta-like variables (deg)
        // (Skip if it's cos(theta), since that's handled above)
        if (b.find("theta") != std::string::npos && b.find("cos") == std::string::npos) {
            hardMin = 0.0;
            hardMax = 180.0;
            minWidth = 1.0;
        }

        // Robust quantile ranges for truth and reco, then take a shared union range
        auto tr = quantile_range(vtrue, /*qlo=*/0.01, /*qhi=*/0.99, /*padFrac=*/0.10,
                                hardMin, hardMax, minWidth);
        auto pr = quantile_range(vreco, /*qlo=*/0.01, /*qhi=*/0.99, /*padFrac=*/0.10,
                                hardMin, hardMax, minWidth);

        double xmin = std::min(tr.first, pr.first);
        double xmax = std::max(tr.second, pr.second);

        // Final safety
        xmin = std::max(xmin, hardMin);
        xmax = std::min(xmax, hardMax);
        if (!(xmax > xmin)) { xmin -= 1.0; xmax += 1.0; }


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
void eval_model(
    const char* filename,
    bool beam_mode=false,
    double wandb_runtime_hours=-1.0,
    const char* output_dir = ".",
    const char* png_path = "",
    int png_width = 0,
    int png_height = 0,
    const char* root_output_name = "combined_output.root"
    ) {

     // Put ROOT into batch mode so canvases are not shown on screen
    Bool_t oldBatch = gROOT->IsBatch();
    gROOT->SetBatch(kTRUE);

    std::string original_dir = gSystem->WorkingDirectory();
    if (output_dir && std::string(output_dir).size() > 0) {
        gSystem->mkdir(output_dir, kTRUE);
        gSystem->ChangeDirectory(output_dir);
    }

    std::string filename_abs = filename ? filename : "";
    if (!filename_abs.empty() && filename_abs.front() != '/') {
        filename_abs = original_dir + "/" + filename_abs;
    }

    std::ifstream infile(filename_abs);
    if (!infile.is_open()) {
        std::cerr << "Could not open file: " << filename_abs << std::endl;
        gSystem->ChangeDirectory(original_dir.c_str());
        gROOT->SetBatch(oldBatch); // restore batch state before returning
        return;
    }

    // === Parse header ===
    std::string line;
    if (!std::getline(infile, line)) {
        std::cerr << "Empty file or missing header: " << filename_abs << std::endl;
        gSystem->ChangeDirectory(original_dir.c_str());
        gROOT->SetBatch(oldBatch);
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

    // --- Flags: does the CSV explicitly contain CosTheta columns? ---
    const bool CSV_HAS_PRED_COS = (colIndex.find("pred_Nu_CosTheta") != colIndex.end());
    const bool CSV_HAS_TRUE_COS = (colIndex.find("true_Nu_CosTheta") != colIndex.end());

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
    std::string root_output_path = root_output_name ? root_output_name : "";
    if (root_output_path.empty()) {
        root_output_path = "combined_output.root";
    }

    Long_t id = 0;
    Long_t size = 0;
    Long_t flags = 0;
    Long_t modtime = 0;
    bool fileExists = (gSystem->GetPathInfo(root_output_path.c_str(), &id, &size, &flags, &modtime) == 0);
    bool hasContents = fileExists && size > 0;

    TFile* outfile = nullptr;
    if (hasContents) {
        outfile = TFile::Open(root_output_path.c_str(), "UPDATE");
        std::cout << "Opened existing " << root_output_path << " in UPDATE mode.\n";
    } else {
        outfile = TFile::Open(root_output_path.c_str(), "RECREATE");
        std::cout << "Created new " << root_output_path << ".\n";
    }

    if (!outfile || outfile->IsZombie()) {
         std::cerr << "Error opening " << root_output_path << std::endl;
         gSystem->ChangeDirectory(original_dir.c_str());
         gROOT->SetBatch(oldBatch);
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

    //lambda that allows for multiple variations of Nu_CosTheta headers
    auto firstExistingKey = [&](const std::vector<std::string>& keys) -> std::string {
        for (const auto& k : keys) {
            if (data.find(k) != data.end()) return k;
        }
        return "";
    };

    std::vector<double> Cos_Theta_nu_pred, Theta_nu_pred;
    std::vector<double> Cos_Theta_nu_true, Theta_nu_true;
    std::vector<double> pred_Mass_squared, true_Mass_squared;
    std::vector<double> pred_baseline, true_baseline;
    std::vector<double> pred_beam_Mass_squared, true_beam_Mass_squared;

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

    }
    infile.close();

    // === Derive Energy from momentum if Energy columns are missing ===
    // Only do this when Px/Py/Pz exist and Energy is absent.
    bool has_trueE_col = (data.find("true_Nu_Energy") != data.end());
    bool has_predE_col = (data.find("pred_Nu_Energy") != data.end());

    bool has_trueMomCols = (data.find("true_Nu_Mom_X") != data.end() &&
                            data.find("true_Nu_Mom_Y") != data.end() &&
                            data.find("true_Nu_Mom_Z") != data.end());

    bool has_predMomCols = (data.find("pred_Nu_Mom_X") != data.end() &&
                            data.find("pred_Nu_Mom_Y") != data.end() &&
                            data.find("pred_Nu_Mom_Z") != data.end());

    // Build derived vectors (only for the missing ones)
    if (!has_trueE_col && has_trueMomCols) {
        const auto& tx = data["true_Nu_Mom_X"];
        const auto& ty = data["true_Nu_Mom_Y"];
        const auto& tz = data["true_Nu_Mom_Z"];
        size_t N = std::min({tx.size(), ty.size(), tz.size()});

        std::vector<double> Etrue;
        Etrue.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            Etrue.push_back(calcEnergyFromP(tx[i], ty[i], tz[i]));
        }

        data["true_Nu_Energy"] = std::move(Etrue);
        std::cout << "[INFO] Derived true_Nu_Energy from true_Nu_Mom_X/Y/Z (massless approx).\n";
    }

    if (!has_predE_col && has_predMomCols) {
        const auto& px = data["pred_Nu_Mom_X"];
        const auto& py = data["pred_Nu_Mom_Y"];
        const auto& pz = data["pred_Nu_Mom_Z"];
        size_t N = std::min({px.size(), py.size(), pz.size()});

        std::vector<double> Epred;
        Epred.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            Epred.push_back(calcEnergyFromP(px[i], py[i], pz[i]));
        }

        data["pred_Nu_Energy"] = std::move(Epred);
        std::cout << "[INFO] Derived pred_Nu_Energy from pred_Nu_Mom_X/Y/Z (massless approx).\n";
    }

    // === Align data vectors and compute derived kinematic quantities ===
    auto hasKey = [&](const std::string& key) {
        return data.find(key) != data.end();
    };

    size_t n_rows = 0;
    for (const auto& kv : data) {
        n_rows = std::max(n_rows, kv.second.size());
    }

    auto ensure_size = [&](std::vector<double>& v, size_t n) {
        if (v.size() < n) v.resize(n, NAN);
    };

    for (auto& kv : data) {
        ensure_size(kv.second, n_rows);
    }

    Cos_Theta_nu_pred.assign(n_rows, NAN);
    Theta_nu_pred.assign(n_rows, NAN);
    Cos_Theta_nu_true.assign(n_rows, NAN);
    Theta_nu_true.assign(n_rows, NAN);
    pred_baseline.assign(n_rows, NAN);
    true_baseline.assign(n_rows, NAN);
    pred_Mass_squared.assign(n_rows, NAN);
    true_Mass_squared.assign(n_rows, NAN);
    pred_beam_Mass_squared.assign(n_rows, NAN);
    true_beam_Mass_squared.assign(n_rows, NAN);

    const bool has_predMom = hasKey("pred_Nu_Mom_X") && hasKey("pred_Nu_Mom_Y") && hasKey("pred_Nu_Mom_Z");
    const bool has_trueMom = hasKey("true_Nu_Mom_X") && hasKey("true_Nu_Mom_Y") && hasKey("true_Nu_Mom_Z");
    const bool has_predCos = hasKey("pred_Nu_CosTheta");
    const bool has_trueCos = hasKey("true_Nu_CosTheta");

    for (size_t i = 0; i < n_rows; ++i) {
        if (has_predMom) {
            double px = data["pred_Nu_Mom_X"][i];
            double py = data["pred_Nu_Mom_Y"][i];
            double pz = data["pred_Nu_Mom_Z"][i];
            Cos_Theta_nu_pred[i] = calcCosTheta(px, py, pz);
            Theta_nu_pred[i] = calcTheta(px, py, pz);
            pred_baseline[i] = calc_baseline(px, py, pz);
        } else if (has_predCos) {
            double c = data["pred_Nu_CosTheta"][i];
            if (std::isfinite(c)) {
                Cos_Theta_nu_pred[i] = c;
                Theta_nu_pred[i] = thetaFromCos(c);
            }
        }

        if (has_trueMom) {
            double tx = data["true_Nu_Mom_X"][i];
            double ty = data["true_Nu_Mom_Y"][i];
            double tz = data["true_Nu_Mom_Z"][i];
            Cos_Theta_nu_true[i] = calcCosTheta(tx, ty, tz);
            Theta_nu_true[i] = calcTheta(tx, ty, tz);
            true_baseline[i] = calc_baseline(tx, ty, tz);
        } else if (has_trueCos) {
            double c = data["true_Nu_CosTheta"][i];
            if (std::isfinite(c)) {
                Cos_Theta_nu_true[i] = c;
                Theta_nu_true[i] = thetaFromCos(c);
            }
        }

        if (hasKey("true_Nu_Energy") && has_trueMom) {
            double E = data["true_Nu_Energy"][i];
            double px = data["true_Nu_Mom_X"][i];
            double py = data["true_Nu_Mom_Y"][i];
            double pz = data["true_Nu_Mom_Z"][i];
            if (std::isfinite(E) && std::isfinite(px) && std::isfinite(py) && std::isfinite(pz)) {
                true_Mass_squared[i] = E*E - (px*px + py*py + pz*pz);
            }
        }

        if (hasKey("pred_Nu_Energy") && has_predMom) {
            double E = data["pred_Nu_Energy"][i];
            double px = data["pred_Nu_Mom_X"][i];
            double py = data["pred_Nu_Mom_Y"][i];
            double pz = data["pred_Nu_Mom_Z"][i];
            if (std::isfinite(E) && std::isfinite(px) && std::isfinite(py) && std::isfinite(pz)) {
                pred_Mass_squared[i] = E*E - (px*px + py*py + pz*pz);
            }
        }
    }

    // Ensure derived cos(theta) columns exist in data when the CSV omits them
    if (!CSV_HAS_PRED_COS) {
        data["pred_Nu_CosTheta"] = Cos_Theta_nu_pred;
    }
    if (!CSV_HAS_TRUE_COS) {
        data["true_Nu_CosTheta"] = Cos_Theta_nu_true;
    }

    // === Beam-only Mass^2 from E and (Theta or CosTheta) when no momentum is present ===
    // m^2 = (E)^2 * (1 - Cos^2(90 - Theta))
    //
    // If CosTheta is available, this simplifies numerically to:
    // Cos(90-Theta) = sin(Theta) => 1 - sin^2(Theta) = cos^2(Theta) => m^2 = E^2 * CosTheta^2
    {
        auto pickKey = [&](const std::initializer_list<std::string>& candidates) -> std::string {
            for (const auto& k : candidates) {
                if (data.find(k) != data.end()) return k;
            }
            return "";
        };

        // Energy keys (your CSV uses these exact names)
        const bool has_trueE = (data.find("true_Nu_Energy") != data.end());
        const bool has_predE = (data.find("pred_Nu_Energy") != data.end());

        // Momentum presence (must be absent for this beam-only definition)
        const bool has_trueMom = (data.find("true_Nu_Mom_X") != data.end() &&
                                data.find("true_Nu_Mom_Y") != data.end() &&
                                data.find("true_Nu_Mom_Z") != data.end());
        const bool has_predMom = (data.find("pred_Nu_Mom_X") != data.end() &&
                                data.find("pred_Nu_Mom_Y") != data.end() &&
                                data.find("pred_Nu_Mom_Z") != data.end());

        // Theta / CosTheta keys (support a couple naming variants)
        const std::string trueThetaKey = pickKey({"true_Nu_Theta", "true_Theta"});
        const std::string predThetaKey = pickKey({"pred_Nu_Theta", "pred_Theta"});

        const std::string trueCosKey   = pickKey({"true_Nu_CosTheta", "true_CosTheta"});
        const std::string predCosKey   = pickKey({"pred_Nu_CosTheta", "pred_CosTheta"});

        const bool has_trueTheta = !trueThetaKey.empty();
        const bool has_predTheta = !predThetaKey.empty();
        const bool has_trueCos   = !trueCosKey.empty();
        const bool has_predCos   = !predCosKey.empty();

        // Only do this for beam-mode trainings with E and Theta/CosTheta but without momentum xyz.
        if (beam_mode && has_trueE && has_predE && !has_trueMom && !has_predMom &&
            ( (has_trueTheta || has_trueCos) && (has_predTheta || has_predCos) ))
        {
            const auto& Etrue = data["true_Nu_Energy"];
            const auto& Epred = data["pred_Nu_Energy"];
            const double kDeg = M_PI / 180.0;

            for (size_t i = 0; i < n_rows; ++i) {
                double m2_true = NAN;
                double m2_pred = NAN;

                if (i < Etrue.size()) {
                    double E = Etrue[i];
                    if (std::isfinite(E)) {
                        if (has_trueCos && i < data[trueCosKey].size()) {
                            double c = data[trueCosKey][i];
                            if (std::isfinite(c)) m2_true = E*E * (c*c);
                        } else if (has_trueTheta && i < data[trueThetaKey].size()) {
                            double th = data[trueThetaKey][i];
                            if (std::isfinite(th)) {
                                double ca = std::cos((90.0 - th) * kDeg);
                                m2_true = E*E * (1.0 - ca*ca);
                            }
                        }
                    }
                }

                if (i < Epred.size()) {
                    double E = Epred[i];
                    if (std::isfinite(E)) {
                        if (has_predCos && i < data[predCosKey].size()) {
                            double c = data[predCosKey][i];
                            if (std::isfinite(c)) m2_pred = E*E * (c*c);
                        } else if (has_predTheta && i < data[predThetaKey].size()) {
                            double th = data[predThetaKey][i];
                            if (std::isfinite(th)) {
                                double ca = std::cos((90.0 - th) * kDeg);
                                m2_pred = E*E * (1.0 - ca*ca);
                            }
                        }
                    }
                }

                true_beam_Mass_squared[i] = m2_true;
                pred_beam_Mass_squared[i] = m2_pred;
            }

            std::cout << "[INFO] Computed beam mass^2 from E and Theta/CosTheta (no momentum columns).\n";
        }
    }

    // === Unbinned TTree (outside plot directories) ===
    outfile->cd();
    TTree* unbinnedTree = dynamic_cast<TTree*>(outfile->Get("unbinned_kinematics"));
    if (!unbinnedTree) {
        unbinnedTree = new TTree("unbinned_kinematics", "Unbinned kinematic quantities per event");
    }

    std::string model_name = dirName;
    if (!unbinnedTree->GetBranch("model_name")) {
        unbinnedTree->Branch("model_name", &model_name);
    } else {
        unbinnedTree->SetBranchAddress("model_name", &model_name);
    }

    std::map<std::string, std::vector<double>> unbinned_data = data;
    unbinned_data["Cos_Theta_nu_pred"] = Cos_Theta_nu_pred;
    unbinned_data["Theta_nu_pred"] = Theta_nu_pred;
    unbinned_data["Cos_Theta_nu_true"] = Cos_Theta_nu_true;
    unbinned_data["Theta_nu_true"] = Theta_nu_true;
    unbinned_data["pred_baseline"] = pred_baseline;
    unbinned_data["true_baseline"] = true_baseline;
    unbinned_data["pred_Mass_squared"] = pred_Mass_squared;
    unbinned_data["true_Mass_squared"] = true_Mass_squared;
    unbinned_data["pred_beam_Mass_squared"] = pred_beam_Mass_squared;
    unbinned_data["true_beam_Mass_squared"] = true_beam_Mass_squared;

    std::map<std::string, double> branch_values;
    for (auto& kv : unbinned_data) {
        ensure_size(kv.second, n_rows);
        if (!unbinnedTree->GetBranch(kv.first.c_str())) {
            unbinnedTree->Branch(kv.first.c_str(), &branch_values[kv.first]);
        } else {
            unbinnedTree->SetBranchAddress(kv.first.c_str(), &branch_values[kv.first]);
        }
    }

    for (size_t i = 0; i < n_rows; ++i) {
        model_name = dirName;
        for (const auto& kv : unbinned_data) {
            const auto& vec = kv.second;
            double val = (i < vec.size()) ? vec[i] : NAN;
            branch_values[kv.first] = val;
        }
        unbinnedTree->Fill();
    }

    unbinnedTree->Write("", TObject::kOverwrite);

    // === Create 1D histograms ===
auto make_hist = [&](const std::string& name, const std::vector<double>& d, double xmin, double xmax) {
    if (d.empty()) return;
    bool has_finite = std::any_of(d.begin(), d.end(), [](double v) { return std::isfinite(v); });
    if (!has_finite) return;
    TH1D* h = new TH1D(name.c_str(), name.c_str(), 500, xmin, xmax);
    h->SetDirectory(0);  // prevent ROOT from auto-managing this hist
    for (double val : d) {
        if (std::isfinite(val)) h->Fill(val);
    }
    if (plotDir) plotDir->cd();
    h->Write();
    std::cout << "Writing hist to: " << gDirectory->GetPath() << " / " << name << std::endl;
};

    auto find_range = [](const std::vector<double>& d) {
        double minVal = std::numeric_limits<double>::infinity();
        double maxVal = -std::numeric_limits<double>::infinity();
        for (double v : d) {
            if (!std::isfinite(v)) continue;
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }
        if (!std::isfinite(minVal) || !std::isfinite(maxVal)) {
            return std::make_pair(0.0, 1.0);
        }
        double margin = 0.05 * std::max(std::abs(minVal), std::abs(maxVal));
        return std::make_pair(minVal - margin, maxVal + margin);
    };

    // 1D histograms for all non true_/pred_ variables (auto range)
    for (const auto& kv : data) {
        if (kv.second.empty()) continue;

        const std::string& name = kv.first;
        // Skip true_*/pred_* here; they will be handled in paired logic below
        if (name.rfind("true_", 0) == 0 || name.rfind("pred_", 0) == 0) continue;

        auto r = find_range(kv.second);
        make_hist(name, kv.second, r.first, r.second);
    }

    // 1D histograms for true/pred pairs with shared axes (range from pred)
    for (const auto& kv : data) {
        const std::string& name = kv.first;

        // Only iterate over predicted variables
        if (name.rfind("pred_", 0) != 0) continue;

        std::string base = name.substr(5);              // strip "pred_"
        std::string trueName = "true_" + base;

        auto itTrue = data.find(trueName);
        if (itTrue == data.end()) continue;             // no matching true_*

        const auto& pred_vals = kv.second;
        const auto& true_vals = itTrue->second;
        if (pred_vals.empty() || true_vals.empty()) continue;

        // Determine axis range from *predicted* values
        auto r = find_range(pred_vals);

        // Predicted histogram
        make_hist(name,     pred_vals, r.first, r.second);
        // True histogram using the SAME range & binning
        make_hist(trueName, true_vals, r.first, r.second);
    }

    // Calculated neutrino mass squared for trainings with 4-momentum 
     if (!true_Mass_squared.empty() || !pred_Mass_squared.empty()) {
    auto r = find_range(pred_Mass_squared);
    make_hist("true_Mass_squared", true_Mass_squared, r.first, r.second);
    make_hist("pred_Mass_squared", pred_Mass_squared, r.first, r.second);
     }

    // Beam-only E+Theta/CosTheta neutrino mass squared (no momentum)
    if (!true_beam_Mass_squared.empty() || !pred_beam_Mass_squared.empty()) {
        auto r = find_range(pred_beam_Mass_squared);
        make_hist("true_beam_Mass_squared", true_beam_Mass_squared, r.first, r.second);
        make_hist("pred_beam_Mass_squared", pred_beam_Mass_squared, r.first, r.second);
    }

    make_hist("Cos_Theta_nu_pred", Cos_Theta_nu_pred, -1, 1);
    make_hist("Theta_nu_pred", Theta_nu_pred, 0, 180);
    make_hist("Cos_Theta_nu_true", Cos_Theta_nu_true, -1, 1);
    make_hist("Theta_nu_true", Theta_nu_true, 0, 180);
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

        const bool cosMode = isCosThetaVar(base);

        for (size_t i = 0; i < true_vals.size(); ++i) {
            double t = true_vals[i];
            double p = pred_vals[i];
            if (!std::isfinite(t) || !std::isfinite(p)) continue;

            if (cosMode) {
                // For cos(theta), use absolute residual: Δcosθ = pred - true
                res.push_back(p - t);
            } else {
                // Default behavior: percent resolution
                if (t != 0.0) {
                    res.push_back(100.0 * (p - t) / t);
                } else {
                    std::cout << "Divide by zero at " << kv.first << " index " << i << std::endl;
                }
            }
        }


        // 1D histogram of resolution / residual for this variable ---
        if (!res.empty()) {
            auto r = find_range(res);

            std::string hname  = "h1_res_" + base;
            std::string htitle;
            if (cosMode) {
                htitle = base + " residual (pred - true);#Delta cos(#theta);Counts";

                // Clamp residual range for readability
                double ymin = std::max(r.first, -0.5);
                double ymax = std::min(r.second,  0.5);
                if (ymin == ymax) { ymin -= 0.01; ymax += 0.01; }
                r = {ymin, ymax};

            } else {
                htitle = base + " percent resolution;Percent resolution (%);Counts";

                // Clamp percent range to ±200% (your existing behavior)
                r = clamp_res_range(r.first, r.second);
            }

            TH1D* hres = new TH1D(hname.c_str(), htitle.c_str(), 200, r.first, r.second);
            hres->SetDirectory(0);

            for (double v : res) {
                if (std::isfinite(v)) hres->Fill(v);
            }

            if (plotDir) plotDir->cd();
            hres->Write();
            std::cout << "Wrote 1D resolution hist: " << hname
                    << " to " << gDirectory->GetPath() << std::endl;
        }

        // 2D resolution plots with rms or std error bars
        graph_resolution_stat(base, true_vals, res, "rms", NUM_BINS, XMIN_DEFAULT, XMAX_DEFAULT, plotDir, !cosMode);
        graph_resolution_stat(base, true_vals, res, "std", NUM_BINS, XMIN_DEFAULT, XMAX_DEFAULT, plotDir, !cosMode);


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

    // store the corresponding truth coordinates for additional 2D plots
    std::vector<double> true_theta_deg;   // true neutrino theta (deg)
    std::vector<double> true_energy_gev;  // true neutrino energy (GeV)

    bool has_trueE = data.find("true_Nu_Energy") != data.end();
    bool has_predE = data.find("pred_Nu_Energy") != data.end();
    bool has_trueMom = (data.find("true_Nu_Mom_X") != data.end()
                        && data.find("true_Nu_Mom_Y") != data.end()
                        && data.find("true_Nu_Mom_Z") != data.end());
    bool has_predMom = (data.find("pred_Nu_Mom_X") != data.end()
                        && data.find("pred_Nu_Mom_Y") != data.end()
                        && data.find("pred_Nu_Mom_Z") != data.end());

    // Accept a few common cosine-theta column name conventions.
    // Add/adjust strings here to match your actual result.csv headers.
    std::string trueCosKey = firstExistingKey({
        "true_Nu_CosTheta",
        "true_Nu_Cos_Theta",
        "true_Cos_Theta_nu",
        "true_Cos_Theta_Nu",
        "true_CosTheta"
    });

    std::string predCosKey = firstExistingKey({
        "pred_Nu_CosTheta",
        "pred_Nu_Cos_Theta",
        "pred_Cos_Theta_nu",
        "pred_Cos_Theta_Nu",
        "pred_CosTheta"
    });

    bool has_costheta = (!trueCosKey.empty() && !predCosKey.empty());

    bool has_theta = (data.find("true_Nu_Theta") != data.end()
                  && data.find("pred_Nu_Theta") != data.end());


    if (has_trueE && has_predE && ((has_trueMom && has_predMom) || has_theta || has_costheta)) {
        // Base sizes from energy columns
        size_t NtrueE  = data["true_Nu_Energy"].size();
        size_t NpredE  = data["pred_Nu_Energy"].size();
        size_t Nmin    = std::min(NtrueE, NpredE);

        // Also constrain by momentum sizes if using momentum
        if (has_trueMom && has_predMom) {
            size_t NtrueMom = data["true_Nu_Mom_X"].size();
            size_t NpredMom = data["pred_Nu_Mom_X"].size();
            Nmin = std::min({Nmin, NtrueMom, NpredMom});
        }

        // Also constrain by theta sizes if using explicit angles
        if (has_theta) {
            size_t NtrueTheta = data["true_Nu_Theta"].size();
            size_t NpredTheta = data["pred_Nu_Theta"].size();
            Nmin = std::min({Nmin, NtrueTheta, NpredTheta});
        }

        // Also constrain by cos(theta) sizes if using cosine columns
        if (has_costheta) {
            size_t NtrueCos = data[trueCosKey].size();
            size_t NpredCos = data[predCosKey].size();
            Nmin = std::min({Nmin, NtrueCos, NpredCos});
        }

        for (size_t i = 0; i < Nmin; ++i) {
            double Etrue = data["true_Nu_Energy"][i];
            double Epred = data["pred_Nu_Energy"][i];

            if (std::isnan(Etrue) || std::isnan(Epred)) continue;
            if (Etrue == 0) continue; // avoid divide by zero

            double thet_true = NAN;
            double thet_pred = NAN;

            if (has_theta) {
                // Use explicit theta columns if present
                thet_true = data["true_Nu_Theta"][i];
                thet_pred = data["pred_Nu_Theta"][i];

            } else if (has_trueMom && has_predMom) {
                // Fall back to momentum-based computation
                double tx = data["true_Nu_Mom_X"][i];
                double ty = data["true_Nu_Mom_Y"][i];
                double tz = data["true_Nu_Mom_Z"][i];
                double px = data["pred_Nu_Mom_X"][i];
                double py = data["pred_Nu_Mom_Y"][i];
                double pz = data["pred_Nu_Mom_Z"][i];

                thet_true = calcTheta(tx, ty, tz);
                thet_pred = calcTheta(px, py, pz);

            } else if (has_costheta) {
                // Compute theta from cos(theta) columns
                double ctrue = data[trueCosKey][i];
                double cpred = data[predCosKey][i];

                thet_true = thetaFromCos(ctrue);
                thet_pred = thetaFromCos(cpred);

            } else {
                // Should not happen given the guards
                continue;
            }

            double eres = 100.0 * (Epred - Etrue) / Etrue;
            double tdiff = thet_pred - thet_true;

            if (std::isfinite(eres) && std::isfinite(tdiff) && std::isfinite(thet_true) && std::isfinite(Etrue)) {
                energy_res_percent.push_back(eres);
                theta_diff_signed.push_back(tdiff);

                // truth coordinates for additional plots
                true_theta_deg.push_back(thet_true);
                true_energy_gev.push_back(Etrue);
            }
        }
    } else {
    std::cout << "Skipping 2D energy/theta plot: missing columns (need true_Nu_Energy, pred_Nu_Energy, and one of true/pred_Nu_Mom_*, true/pred_Nu_Theta, true/pred_Nu_CosTheta)." << std::endl;
    }

    auto draw2DWithContours = [&](const std::string& cname,
                                const std::string& hname,
                                const std::string& title,
                                const std::vector<double>& x,
                                const std::vector<double>& y,
                                int nbx, double xmin, double xmax,
                                int nby, double ymin, double ymax)
    {
        if (x.empty() || y.empty() || x.size() != y.size()) return;

        TCanvas* c = new TCanvas(cname.c_str(), cname.c_str(), 900, 800);
        gStyle->SetOptStat(0);

        TH2D* h2 = new TH2D(hname.c_str(), title.c_str(), nbx, xmin, xmax, nby, ymin, ymax);
        h2->SetDirectory(0);

        for (size_t i = 0; i < x.size(); ++i) {
            if (std::isfinite(x[i]) && std::isfinite(y[i])) h2->Fill(x[i], y[i]);
        }

        c->cd();
        h2->Draw("COLZ");

        // Coarsen + smooth for stable contours (same approach as main plot)
        TH2D* hcont = (TH2D*)h2->Clone((std::string("hcont_") + hname).c_str());
        hcont->SetDirectory(0);
        hcont->Rebin2D(2, 2);
        hcont->Smooth(1);

        std::vector<double> levels = calcLevels(hcont, {0.95, 0.90, 0.68});
        if (levels.size() == 3) {
            drawLargestContourAtLevel(hcont, levels[0], kRed+1,    3);
            drawLargestContourAtLevel(hcont, levels[1], kOrange+7, 3);
            drawLargestContourAtLevel(hcont, levels[2], kGreen+2,  3);
        } else {
            for (size_t i = 0; i < levels.size(); ++i)
                drawLargestContourAtLevel(hcont, levels[i], kRed + (int)i, 3);
        }

        if (plotDir) plotDir->cd();
        c->Write((cname + "_canvas").c_str());
        h2->Write(hname.c_str());
    };

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
        
        double ymin = beam_mode ? -0.5 : -180.0;
        double ymax = beam_mode ?  0.5 :  180.0;

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


        // Draw ellipse centered at (0,0) with:
        // x: Energy resolution (%) semi-axis (±10% -> a=10)
        // y: Delta-theta (deg) semi-axis:
        //   - normal mode: ±30 deg -> b=30
        //   - beam_mode:   ±1  deg -> b=1
        double ell_a = 10.0;
        double ell_b = beam_mode ? 0.1 : 30.0;

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
        updateEllipseCSV(modelName, ell_a, ell_b, frac, wandb_runtime_hours);

        // Write canvas and histogram to the selected directory
        if (plotDir) plotDir->cd();
        c2->Write("energy_theta_2d_canvas");
        if (png_path && std::string(png_path).size() > 0) {
            if (png_width > 0 && png_height > 0) {
                c2->SetCanvasSize(png_width, png_height);
                c2->Update();
            }
            c2->SaveAs(png_path);
        }
        h2->Write("h2_energy_theta");

        // ============================================================
        // Secondary "zoomed" plot (dynamic range) for detailed contours
        // Does NOT compute ellipse fraction / does NOT update CSV
        // ============================================================
        {
            // Robust ranges using quantiles (ignore extreme tails)
            // Energy resolution: keep your ±200% hard clamp, but zoom to the core
            auto xr = quantile_range(energy_res_percent,
                                    /*qlo=*/0.01, /*qhi=*/0.99,
                                    /*padFrac=*/0.10,
                                    /*hardMin=*/-200.0, /*hardMax=*/200.0,
                                    /*minWidth=*/5.0);

            // Theta difference: zoom based on the actual distribution
            // For beam trainings, this will often shrink far below ±0.5 deg.
            auto yr = quantile_range(theta_diff_signed,
                                    /*qlo=*/0.01, /*qhi=*/0.99,
                                    /*padFrac=*/0.10,
                                    /*hardMin=*/-180.0, /*hardMax=*/180.0,
                                    /*minWidth=*/0.02);

            // Make Δθ symmetric about 0 for nicer interpretation
            auto ys = symmetric_about_zero(yr.first, yr.second,
                                        /*minHalfWidth=*/(beam_mode ? 0.01 : 1.0));
            double ymin_zoom = ys.first;
            double ymax_zoom = ys.second;

            double xmin_zoom = xr.first;
            double xmax_zoom = xr.second;

            TCanvas* c2z = new TCanvas("energy_theta_2d_zoom", "Energy% vs DeltaTheta (zoom)", 900, 800);
            gStyle->SetOptStat(0);

            TH2D* h2z = new TH2D("h2_energy_theta_zoom",
                                "Energy Resolution (%) vs #Delta#theta (deg) [zoomed];Energy Resolution (%);#Delta#theta (deg)",
                                300, xmin_zoom, xmax_zoom,
                                300, ymin_zoom, ymax_zoom);
            h2z->SetDirectory(0);

            for (size_t i = 0; i < energy_res_percent.size(); ++i) {
                h2z->Fill(energy_res_percent[i], theta_diff_signed[i]);
            }

            c2z->cd();
            h2z->Draw("COLZ");

            // Same contour workflow as main plot
            TH2D* hcontz = (TH2D*)h2z->Clone("hcont_energy_theta_zoom");
            hcontz->SetDirectory(0);
            hcontz->Rebin2D(2, 2);
            hcontz->Smooth(1);

            std::vector<double> levelsZ = calcLevels(hcontz, {0.95, 0.90, 0.68});

            // Draw largest closed loop for each level (reuse your helper)
            // NOTE: choose distinct colors/widths if you want; here I mirror your main logic structure.
            drawLargestContourAtLevel(hcontz, levelsZ[0], kBlue+1, 2);
            drawLargestContourAtLevel(hcontz, levelsZ[1], kGreen+2, 2);
            drawLargestContourAtLevel(hcontz, levelsZ[2], kRed+1, 2);

            // Write zoom products to the same TDirectory
            if (plotDir) plotDir->cd();
            c2z->Write("energy_theta_2d_zoom_canvas");
            h2z->Write("h2_energy_theta_zoom");

            // Clean up heap objects you created here (optional in ROOT macro, but good hygiene)
            delete hcontz;
            delete h2z;
            delete c2z;
        }

        // =======================
        // additional 2D plots
        // =======================

        // Common Y ranges (match conventions used in main plot)
        double tdiff_ymin = beam_mode ? -0.5 : -180.0;
        double tdiff_ymax = beam_mode ?  0.5 :  180.0;

        // Energy resolution range (use your existing clamp_res_range)
        double eres_ymin = *std::min_element(energy_res_percent.begin(), energy_res_percent.end());
        double eres_ymax = *std::max_element(energy_res_percent.begin(), energy_res_percent.end());
        auto eres_rng = clamp_res_range(eres_ymin, eres_ymax);
        eres_ymin = eres_rng.first;
        eres_ymax = eres_rng.second;

        // True energy X range from data with padding
        double E_xmin = *std::min_element(true_energy_gev.begin(), true_energy_gev.end());
        double E_xmax = *std::max_element(true_energy_gev.begin(), true_energy_gev.end());
        double E_xpad = 0.05 * std::max(1.0, std::fabs(E_xmax - E_xmin));
        E_xmin -= E_xpad;
        E_xmax += E_xpad;
        if (E_xmin == E_xmax) { E_xmin -= 1; E_xmax += 1; }

        // True theta X range is physically bounded
        double Th_xmin = 0.0;
        double Th_xmax = 180.0;

        // (1) theta difference vs true neutrino theta
        draw2DWithContours(
            "thetaDiff_vs_trueTheta",
            "h2_thetaDiff_vs_trueTheta",
            "#Delta#theta (deg) vs true #theta_{#nu};true #theta_{#nu} (deg);#Delta#theta (deg)",
            true_theta_deg, theta_diff_signed,
            200, Th_xmin, Th_xmax,
            200, tdiff_ymin, tdiff_ymax
        );

        // (2) energy resolution vs true neutrino theta
        draw2DWithContours(
            "energyRes_vs_trueTheta",
            "h2_energyRes_vs_trueTheta",
            "Energy resolution (%) vs true #theta_{#nu};true #theta_{#nu} (deg);Energy resolution (%)",
            true_theta_deg, energy_res_percent,
            200, Th_xmin, Th_xmax,
            200, eres_ymin, eres_ymax
        );

        // (3) theta difference vs true neutrino energy
        draw2DWithContours(
            "thetaDiff_vs_trueEnergy",
            "h2_thetaDiff_vs_trueEnergy",
            "#Delta#theta (deg) vs true E_{#nu};true E_{#nu} (GeV);#Delta#theta (deg)",
            true_energy_gev, theta_diff_signed,
            200, E_xmin, E_xmax,
            200, tdiff_ymin, tdiff_ymax
        );

        // (4) energy resolution vs true neutrino energy
        draw2DWithContours(
            "energyRes_vs_trueEnergy",
            "h2_energyRes_vs_trueEnergy",
            "Energy resolution (%) vs true E_{#nu};true E_{#nu} (GeV);Energy resolution (%)",
            true_energy_gev, energy_res_percent,
            200, E_xmin, E_xmax,
            200, eres_ymin, eres_ymax
        );


    } else {
        std::cout << "No 2D entries available; energy/theta 2D histogram not created." << std::endl;
    }

    // Finish writing output file (preserve existing behavior)
    outfile->cd();
    outfile->Close();
    std::cout << "All outputs written to " << root_output_path << " in directory: " 
              << dirName << std::endl;
    
    // Restore previous batch mode
    gROOT->SetBatch(oldBatch);

}
