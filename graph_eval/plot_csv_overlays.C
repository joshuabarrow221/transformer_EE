// plot_csv_overlays.C
// General ROOT macro to compare up to 8 CSV distributions with shared styling.
//
// Plot spec format (semicolon-separated list, max 8):
//   file.csv|column_name|legend label|color|line_style|line_width
// Example:
//   "/tmp/a.csv|E_MAPE|SV AR23 Natural|kRed|1|2;/tmp/b.csv|E_MAPE|SV G2111a Natural|kBlue|2|2"
//
// CLI usage:
// root -l -b -q 'plot_csv_overlays.C("spec1;spec2",160,-4,4,true,-1,-1,true,0.5,
//   "Energy Resolution (%)","Normalized Events","DUNE ND comparison","",
//   "combined_output.root","csv_overlays","energy_overlay","energy_overlay.png")'

#include <TCanvas.h>
#include <TColor.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TLine.h>
#include <TROOT.h>
#include <TStyle.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct PlotSpec {
  std::string file_path;
  std::string column_name;
  std::string legend_label;
  int color = kBlack;
  int line_style = 1;
  int line_width = 2;
};

static inline std::string trim(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) e--;
  return s.substr(b, e - b);
}

static std::vector<std::string> split(const std::string& in, char delim) {
  std::vector<std::string> out;
  std::stringstream ss(in);
  std::string token;
  while (std::getline(ss, token, delim)) out.push_back(trim(token));
  return out;
}

static std::vector<std::string> split_csv_simple(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  cur.reserve(line.size());
  bool in_quotes = false;

  for (char ch : line) {
    if (ch == '"') {
      in_quotes = !in_quotes;
      continue;
    }
    if (ch == ',' && !in_quotes) {
      out.push_back(trim(cur));
      cur.clear();
    } else {
      cur.push_back(ch);
    }
  }
  out.push_back(trim(cur));
  return out;
}

static bool parse_double(const std::string& s, double& out) {
  std::string t = trim(s);
  if (t.empty()) return false;
  char* end = nullptr;
  out = std::strtod(t.c_str(), &end);
  if (end == t.c_str() || *end != '\0') return false;
  return std::isfinite(out);
}

static int parse_color(const std::string& token) {
  std::string t = trim(token);
  if (t.empty()) return kBlack;

  static const std::map<std::string, int> kMap = {
      {"kBlack", kBlack},     {"kRed", kRed},         {"kBlue", kBlue},
      {"kGreen", kGreen},     {"kMagenta", kMagenta}, {"kCyan", kCyan},
      {"kOrange", kOrange},   {"kViolet", kViolet},   {"kGray", kGray},
      {"black", kBlack},      {"red", kRed},          {"blue", kBlue},
      {"green", kGreen},      {"magenta", kMagenta},  {"cyan", kCyan},
      {"orange", kOrange},    {"violet", kViolet},    {"gray", kGray},
  };

  auto it = kMap.find(t);
  if (it != kMap.end()) return it->second;

  double x = 0.0;
  if (parse_double(t, x)) return static_cast<int>(x);

  std::cerr << "[WARNING] Unknown color token '" << t << "'. Using kBlack.\n";
  return kBlack;
}

static PlotSpec parse_plot_spec(const std::string& one_spec) {
  auto parts = split(one_spec, '|');
  if (parts.size() < 3) {
    throw std::runtime_error(
        "Each plot spec must have at least 3 fields: file|column|label");
  }

  PlotSpec p;
  p.file_path = parts[0];
  p.column_name = parts[1];
  p.legend_label = parts[2];

  if (parts.size() > 3 && !parts[3].empty()) p.color = parse_color(parts[3]);
  if (parts.size() > 4 && !parts[4].empty()) p.line_style = std::stoi(parts[4]);
  if (parts.size() > 5 && !parts[5].empty()) p.line_width = std::stoi(parts[5]);

  return p;
}

static std::vector<PlotSpec> parse_plot_specs(const std::string& specs) {
  auto chunks = split(specs, ';');
  std::vector<PlotSpec> out;
  out.reserve(chunks.size());

  for (const auto& c : chunks) {
    if (!c.empty()) out.push_back(parse_plot_spec(c));
  }

  if (out.empty()) {
    throw std::runtime_error("No plot specs were provided.");
  }
  if (out.size() > 8) {
    throw std::runtime_error("At most 8 plot specs are allowed per call.");
  }
  return out;
}

static std::vector<double> load_numeric_column(const std::string& path,
                                               const std::string& col_name) {
  std::ifstream fin(path);
  if (!fin) {
    throw std::runtime_error("Cannot open CSV: " + path);
  }

  std::string line;
  if (!std::getline(fin, line)) {
    throw std::runtime_error("CSV is empty: " + path);
  }

  auto header = split_csv_simple(line);
  int col_idx = -1;
  for (size_t i = 0; i < header.size(); ++i) {
    if (header[i] == col_name) {
      col_idx = static_cast<int>(i);
      break;
    }
  }
  if (col_idx < 0) {
    std::ostringstream oss;
    oss << "Column '" << col_name << "' not found in " << path << ". Header fields:";
    for (const auto& h : header) oss << " " << h;
    throw std::runtime_error(oss.str());
  }

  std::vector<double> values;
  values.reserve(10000);
  size_t row_num = 1;

  while (std::getline(fin, line)) {
    row_num++;
    if (trim(line).empty()) continue;
    auto fields = split_csv_simple(line);
    if (static_cast<int>(fields.size()) <= col_idx) {
      std::cerr << "[WARNING] Skipping malformed row " << row_num << " in " << path
                << " (not enough columns).\n";
      continue;
    }

    double val = 0.0;
    if (!parse_double(fields[col_idx], val)) continue;
    values.push_back(val);
  }

  if (values.empty()) {
    throw std::runtime_error("No numeric values found for column '" + col_name +
                             "' in " + path);
  }
  return values;
}

static TDirectory* get_or_make_dir(TFile* fout, const std::string& path) {
  if (!fout) return nullptr;
  if (path.empty()) return fout;

  std::stringstream ss(path);
  std::string segment;
  TDirectory* current = fout;

  while (std::getline(ss, segment, '/')) {
    segment = trim(segment);
    if (segment.empty()) continue;

    TDirectory* next = dynamic_cast<TDirectory*>(current->Get(segment.c_str()));
    if (!next) next = current->mkdir(segment.c_str());
    current = next;
  }

  return current;
}

}  // namespace

void plot_csv_overlays(
    const char* plot_specs_c,
    int nbins = 160,
    double xmin = -4.0,
    double xmax = 4.0,
    bool normalize = true,
    double ymin = -1.0,
    double ymax = -1.0,
    bool draw_vertical_zero = true,
    double symmetric_vertical = -1.0,
    const char* x_title_c = "Energy Resolution (%)",
    const char* y_title_c = "Normalized Events",
    const char* canvas_title_c = "",
    const char* legend_header_c = "",
    const char* output_root_c = "combined_output.root",
    const char* output_tdirectory_c = "csv_overlay",
    const char* output_canvas_name_c = "overlay_canvas",
    const char* output_png_c = "") {
  try {
    if (!plot_specs_c || std::string(plot_specs_c).empty()) {
      throw std::runtime_error("plot_specs string is required.");
    }
    if (nbins <= 0) throw std::runtime_error("nbins must be > 0");
    if (xmax <= xmin) throw std::runtime_error("xmax must be > xmin");

    const std::string x_title = x_title_c ? x_title_c : "X";
    const std::string y_title = y_title_c ? y_title_c : (normalize ? "Normalized Events" : "Events");
    const std::string canvas_title = canvas_title_c ? canvas_title_c : "";
    const std::string legend_header = legend_header_c ? legend_header_c : "";
    const std::string output_root = output_root_c ? output_root_c : "combined_output.root";
    const std::string output_tdirectory = output_tdirectory_c ? output_tdirectory_c : "csv_overlay";
    const std::string output_canvas_name = output_canvas_name_c ? output_canvas_name_c : "overlay_canvas";
    const std::string output_png = output_png_c ? output_png_c : "";

    auto plot_specs = parse_plot_specs(plot_specs_c);

    gStyle->SetOptStat(0);

    std::vector<std::unique_ptr<TH1D>> hists;
    hists.reserve(plot_specs.size());

    double max_y = 0.0;
    for (size_t i = 0; i < plot_specs.size(); ++i) {
      const auto& spec = plot_specs[i];
      auto values = load_numeric_column(spec.file_path, spec.column_name);

      std::ostringstream hname;
      hname << "h_overlay_" << i;
      auto h = std::make_unique<TH1D>(hname.str().c_str(), "", nbins, xmin, xmax);

      for (double v : values) h->Fill(v);

      if (normalize && h->Integral("width") > 0.0) {
        h->Scale(1.0 / h->Integral("width"));
      }

      h->SetLineColor(spec.color);
      h->SetLineStyle(spec.line_style);
      h->SetLineWidth(spec.line_width);
      h->SetTitle(canvas_title.c_str());
      h->GetXaxis()->SetTitle(x_title.c_str());
      h->GetYaxis()->SetTitle(y_title.c_str());
      h->GetXaxis()->CenterTitle();
      h->GetYaxis()->CenterTitle();

      max_y = std::max(max_y, h->GetMaximum());
      hists.push_back(std::move(h));
    }

    if (max_y <= 0.0) max_y = 1.0;

    auto c = std::make_unique<TCanvas>(output_canvas_name.c_str(), output_canvas_name.c_str(), 1600, 1000);
    c->SetGrid(0, 0);

    if (ymin >= 0.0 || ymax > 0.0) {
      double lo = (ymin >= 0.0) ? ymin : 0.0;
      double hi = (ymax > 0.0) ? ymax : max_y * 1.2;
      hists.front()->SetMinimum(lo);
      hists.front()->SetMaximum(hi);
    } else {
      hists.front()->SetMinimum(0.0);
      hists.front()->SetMaximum(max_y * 1.2);
    }

    hists.front()->Draw("hist");
    for (size_t i = 1; i < hists.size(); ++i) hists[i]->Draw("hist same");

    auto leg = std::make_unique<TLegend>(0.68, 0.66, 0.98, 0.96);
    leg->SetBorderSize(1);
    leg->SetFillStyle(0);
    if (!legend_header.empty()) leg->SetHeader(legend_header.c_str(), "C");
    for (size_t i = 0; i < plot_specs.size(); ++i) {
      leg->AddEntry(hists[i].get(), plot_specs[i].legend_label.c_str(), "l");
    }
    leg->Draw();

    std::vector<std::unique_ptr<TLine>> lines;
    if (draw_vertical_zero) {
      auto line0 = std::make_unique<TLine>(0.0, hists.front()->GetMinimum(), 0.0, hists.front()->GetMaximum());
      line0->SetLineColor(kGray + 2);
      line0->SetLineStyle(2);
      line0->SetLineWidth(2);
      line0->Draw();
      lines.push_back(std::move(line0));
    }

    if (symmetric_vertical > 0.0) {
      auto lp = std::make_unique<TLine>(symmetric_vertical, hists.front()->GetMinimum(), symmetric_vertical,
                                        hists.front()->GetMaximum());
      lp->SetLineColor(kGray + 1);
      lp->SetLineStyle(7);
      lp->SetLineWidth(2);
      lp->Draw();

      auto lm = std::make_unique<TLine>(-symmetric_vertical, hists.front()->GetMinimum(), -symmetric_vertical,
                                        hists.front()->GetMaximum());
      lm->SetLineColor(kGray + 1);
      lm->SetLineStyle(7);
      lm->SetLineWidth(2);
      lm->Draw();

      lines.push_back(std::move(lp));
      lines.push_back(std::move(lm));
    }

    c->Modified();
    c->Update();

    std::unique_ptr<TFile> fout(TFile::Open(output_root.c_str(), "UPDATE"));
    if (!fout || fout->IsZombie()) {
      throw std::runtime_error("Failed to open ROOT output file: " + output_root);
    }

    TDirectory* out_dir = get_or_make_dir(fout.get(), output_tdirectory);
    if (!out_dir) throw std::runtime_error("Failed to create output TDirectory: " + output_tdirectory);
    out_dir->cd();

    c->Write(output_canvas_name.c_str(), TObject::kOverwrite);
    for (const auto& h : hists) h->Write(h->GetName(), TObject::kOverwrite);
    leg->Write((output_canvas_name + std::string("_legend")).c_str(), TObject::kOverwrite);

    fout->Write();
    fout->Close();

    if (!output_png.empty()) c->SaveAs(output_png.c_str());

    std::cout << "[INFO] Wrote canvas + histograms to " << output_root << " under directory '"
              << output_tdirectory << "'.\n";
    if (!output_png.empty()) {
      std::cout << "[INFO] Saved image: " << output_png << "\n";
    }
  } catch (const std::exception& ex) {
    std::cerr << "[ERROR] plot_csv_overlays failed: " << ex.what() << "\n";
    throw;
  }
}
