// eval_model_correlations.C
// Build correlation scatter plots from correlation_inference.py outputs.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TColor.h"
#include "TFile.h"
#include "TGraph.h"
#include "TH2D.h"
#include "TLine.h"
#include "TMarker.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"

namespace {

std::vector<std::string> splitCSVLine(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        out.push_back(token);
    }
    return out;
}

bool toDouble(const std::string& in, double& out) {
    if (in.empty()) return false;
    char* end = nullptr;
    out = std::strtod(in.c_str(), &end);
    return end != in.c_str() && std::isfinite(out);
}

std::map<std::string, std::vector<double>> readNumericColumns(
    const std::string& csvPath,
    const std::vector<std::string>& requiredColumns,
    long maxRows,
    bool& ok
) {
    ok = false;
    std::ifstream fin(csvPath.c_str());
    if (!fin.is_open()) {
        std::cerr << "[ERROR] Could not open CSV: " << csvPath << "\n";
        return {};
    }

    std::string headerLine;
    if (!std::getline(fin, headerLine)) {
        std::cerr << "[ERROR] Empty CSV: " << csvPath << "\n";
        return {};
    }

    std::vector<std::string> headers = splitCSVLine(headerLine);
    std::map<std::string, int> colIndex;
    for (size_t i = 0; i < headers.size(); ++i) colIndex[headers[i]] = static_cast<int>(i);

    for (const auto& column : requiredColumns) {
        if (colIndex.find(column) == colIndex.end()) {
            std::cerr << "[ERROR] Required column not found: " << column << "\n";
            return {};
        }
    }

    std::map<std::string, std::vector<double>> columns;
    for (const auto& c : requiredColumns) columns[c] = {};

    std::string line;
    long rowCount = 0;
    while (std::getline(fin, line)) {
        if (maxRows > 0 && rowCount >= maxRows) break;
        std::vector<std::string> fields = splitCSVLine(line);
        bool rowValid = true;
        std::map<std::string, double> parsed;

        for (const auto& c : requiredColumns) {
            int idx = colIndex[c];
            if (idx < 0 || idx >= static_cast<int>(fields.size())) {
                rowValid = false;
                break;
            }
            double v = 0.0;
            if (!toDouble(fields[idx], v)) {
                rowValid = false;
                break;
            }
            parsed[c] = v;
        }
        if (!rowValid) continue;

        for (const auto& c : requiredColumns) columns[c].push_back(parsed[c]);
        ++rowCount;
    }

    ok = true;
    return columns;
}

void minmax(const std::vector<double>& v, double& lo, double& hi) {
    lo = std::numeric_limits<double>::infinity();
    hi = -std::numeric_limits<double>::infinity();
    for (double x : v) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
    }
}

int paletteColor(double norm01) {
    norm01 = std::max(0.0, std::min(1.0, norm01));
    int n = gStyle->GetNumberContours();
    if (n < 2) n = 255;
    int idx = static_cast<int>(norm01 * (n - 1));
    return TColor::GetColorPalette(idx);
}

}  // namespace

void eval_model_correlations(
    const char* csv_path,
    const char* x_column,
    const char* y_column,
    const char* color_column = "",
    const char* output_dir = ".",
    const char* output_stem = "corr_plot",
    int max_points = -1,
    bool draw_unity_line = true
) {
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);
    gStyle->SetNumberContours(255);

    std::string xcol = x_column;
    std::string ycol = y_column;
    std::string ccol = color_column;

    std::vector<std::string> needed = {xcol, ycol};
    if (!ccol.empty()) needed.push_back(ccol);

    bool ok = false;
    std::map<std::string, std::vector<double>> data = readNumericColumns(
        csv_path,
        needed,
        static_cast<long>(max_points),
        ok
    );
    if (!ok) return;

    const auto& x = data[xcol];
    const auto& y = data[ycol];
    if (x.empty() || y.empty()) {
        std::cerr << "[ERROR] No valid rows loaded for plotting.\n";
        return;
    }

    double xmin, xmax, ymin, ymax;
    minmax(x, xmin, xmax);
    minmax(y, ymin, ymax);

    double xpad = 0.05 * (xmax - xmin + 1e-9);
    double ypad = 0.05 * (ymax - ymin + 1e-9);

    std::string outDir = output_dir;
    gSystem->mkdir(outDir.c_str(), true);
    std::string stem = output_stem;

    TCanvas* canvas = new TCanvas("c_corr", "Correlation plot", 1500, 1100);
    TH2D* frame = new TH2D(
        "frame",
        Form("Correlation; %s; %s", xcol.c_str(), ycol.c_str()),
        100,
        xmin - xpad,
        xmax + xpad,
        100,
        ymin - ypad,
        ymax + ypad
    );
    frame->Draw();

    if (!ccol.empty()) {
        const auto& c = data[ccol];
        double cmin, cmax;
        minmax(c, cmin, cmax);

        for (size_t i = 0; i < x.size(); ++i) {
            double norm = (cmax > cmin) ? (c[i] - cmin) / (cmax - cmin) : 0.5;
            int color = paletteColor(norm);
            TMarker* m = new TMarker(x[i], y[i], 20);
            m->SetMarkerSize(0.45);
            m->SetMarkerColor(color);
            m->Draw("same");
        }

        TH2D* cbar = new TH2D(
            "cbar", Form(";%s;", ccol.c_str()), 1, cmin, cmax, 255, 0.0, 1.0
        );
        for (int ib = 1; ib <= 255; ++ib) cbar->SetBinContent(1, ib, ib);
        TCanvas* cbarCanvas = new TCanvas("c_colorbar", "Colorbar", 450, 1100);
        cbar->Draw("colz");
        cbarCanvas->SaveAs((outDir + "/" + stem + "_colorbar.png").c_str());
        delete cbarCanvas;
    } else {
        TGraph* gr = new TGraph(static_cast<int>(x.size()), x.data(), y.data());
        gr->SetMarkerStyle(20);
        gr->SetMarkerSize(0.5);
        gr->SetMarkerColor(kBlue + 1);
        gr->Draw("P same");
    }

    if (draw_unity_line) {
        double lo = std::max(xmin - xpad, ymin - ypad);
        double hi = std::min(xmax + xpad, ymax + ypad);
        TLine* unity = new TLine(lo, lo, hi, hi);
        unity->SetLineStyle(2);
        unity->SetLineWidth(2);
        unity->Draw("same");
    }

    std::string pngPath = outDir + "/" + stem + ".png";
    canvas->SaveAs(pngPath.c_str());

    std::string rootPath = outDir + "/" + stem + ".root";
    TFile fout(rootPath.c_str(), "RECREATE");
    canvas->Write("corr_canvas");
    fout.Close();

    std::cout << "[INFO] Saved correlation plot: " << pngPath << "\n";
    std::cout << "[INFO] Saved ROOT output: " << rootPath << "\n";

    delete canvas;
}
