// merge_result_csv_two.C
// ROOT helper macro: merge one variable's true/pred columns from a "single-var" result.csv
// into a combined CSV. Ensures true_Topology matches if appending.
//
// Usage:
//   root -l -b -q 'merge_result_csv_two.C("","momx.csv","out.csv","Mom_X")'
//   root -l -b -q 'merge_result_csv_two.C("out.csv","momy.csv","out.csv","Mom_Y")'
//
// Notes:
// - Assumes simple CSV (no quoted commas).
// - Assumes "true_Topology" exists in new file.
// - Variable column mapping is handled in varcols_for().

#include <TSystem.h>
#include <TString.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cctype>

static std::string find_training_header_field(const std::vector<std::string>& header) {
  for (auto const& h : header) {
    if (h.find("NpNpi_") != std::string::npos) return h;
  }
  // fallback: last column name
  if (!header.empty()) return header.back();
  return "";
}

static std::string base_prefix_up_to_NpNpi(const std::string& training_field) {
  auto pos = training_field.find("NpNpi");
  if (pos == std::string::npos) return "";
  return training_field.substr(0, pos + std::string("NpNpi").size());
}

static inline std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace((unsigned char)s[b])) b++;
  while (e > b && std::isspace((unsigned char)s[e - 1])) e--;
  return s.substr(b, e - b);
}

static inline std::vector<std::string> split_csv_simple(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  cur.reserve(line.size());
  for (char ch : line) {
    if (ch == ',') { out.push_back(trim(cur)); cur.clear(); }
    else cur.push_back(ch);
  }
  out.push_back(trim(cur));
  return out;
}

static long double parse_ld(const std::string& s) {
  char* end = nullptr;
  std::string ts = trim(s);
  long double v = std::strtold(ts.c_str(), &end);
  if (end == ts.c_str() || *end != '\0') {
    throw std::runtime_error("Failed to parse numeric: '" + s + "'");
  }
  return v;
}

static bool approx_equal(long double a, long double b, long double rel=1e-12L, long double abs=1e-9L) {
  long double diff = fabsl(a - b);
  if (diff <= abs) return true;
  long double denom = std::max(fabsl(a), fabsl(b));
  return diff <= rel * (denom > 0 ? denom : 1.0L);
}

struct CSVTable {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;
  std::unordered_map<std::string, size_t> idx;
};

static CSVTable read_csv(const std::string& path) {
  std::ifstream fin(path);
  if (!fin) throw std::runtime_error("Cannot open CSV: " + path);

  CSVTable t;
  std::string line;
  if (!std::getline(fin, line)) throw std::runtime_error("Empty CSV: " + path);

  t.header = split_csv_simple(line);
  for (size_t i=0;i<t.header.size();i++) t.idx[t.header[i]] = i;

  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    auto f = split_csv_simple(line);
    if (f.size() != t.header.size()) {
      std::ostringstream oss;
      oss << "CSV row has " << f.size() << " fields but header has " << t.header.size()
          << " in file: " << path;
      throw std::runtime_error(oss.str());
    }
    t.rows.push_back(std::move(f));
  }
  return t;
}

static size_t require_col(const CSVTable& t, const std::string& name, const std::string& file) {
  auto it = t.idx.find(name);
  if (it == t.idx.end()) {
    std::ostringstream oss;
    oss << "Missing required column '" << name << "' in " << file << "\nAvailable columns:\n";
    for (auto const& h : t.header) oss << "  - " << h << "\n";
    throw std::runtime_error(oss.str());
  }
  return it->second;
}

struct VarCols { std::string tcol; std::string pcol; };

static VarCols varcols_for(const std::string& var) {
  // Adjust Energy mapping if needed for your CSV headers.
  if (var == "Energy") return {"true_Nu_Energy", "pred_Nu_Energy"};
  if (var == "Theta")  return {"true_Nu_Theta",  "pred_Nu_Theta"};
  if (var == "Mom_X")  return {"true_Nu_Mom_X",  "pred_Nu_Mom_X"};
  if (var == "Mom_Y")  return {"true_Nu_Mom_Y",  "pred_Nu_Mom_Y"};
  if (var == "Mom_Z")  return {"true_Nu_Mom_Z",  "pred_Nu_Mom_Z"};
  throw std::runtime_error("Unknown var token: " + var);
}

static bool file_exists(const std::string& p) {
  return (gSystem->AccessPathName(p.c_str()) == kFALSE);
}

static void write_csv(const std::string& out,
                      const std::vector<std::string>& header,
                      const std::vector<std::vector<std::string>>& rows) {
  std::ofstream fout(out);
  if (!fout) throw std::runtime_error("Cannot write: " + out);

  for (size_t i=0;i<header.size();i++) {
    fout << header[i];
    if (i+1<header.size()) fout << ",";
  }
  fout << "\n";

  for (auto const& r : rows) {
    for (size_t i=0;i<r.size();i++) {
      fout << r[i];
      if (i+1<r.size()) fout << ",";
    }
    fout << "\n";
  }
}

static void append_warning(const std::string& warnings_file, const std::string& msg) {
  if (warnings_file.empty()) return;
  std::ofstream f(warnings_file, std::ios::app);
  if (f) f << msg << "\n";
}

static bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size()-suf.size(), suf.size(), suf) == 0;
}

static std::string update_label_header_append(const std::string& current_label,
                                              const std::string& tag) {
  const std::string suffix = "_Topology";

  std::string core = current_label;
  if (ends_with(core, suffix)) {
    core = core.substr(0, core.size() - suffix.size());
  }

  // Ensure separator before new tag
  if (core.back() != '_') core += "_";

  const std::string token = "single_var_" + tag;

  // Avoid duplication
  if (core.find(token) == std::string::npos) {
    core += token;
  }

  return core + suffix;
}

void merge_result_csv_two(const char* in_combined_c,
                          const char* in_new_c,
                          const char* out_combined_c,
                          const char* var_token_c,
                          const char* tag_c,
                          const char* warnings_file_c) {
  std::string in_combined = in_combined_c ? std::string(in_combined_c) : "";
  std::string in_new      = in_new_c ? std::string(in_new_c) : "";
  std::string out_combined= out_combined_c ? std::string(out_combined_c) : "";
  std::string var_token   = var_token_c ? std::string(var_token_c) : "";
  std::string tag         = tag_c ? std::string(tag_c) : "";
  std::string warnings_file = warnings_file_c ? std::string(warnings_file_c) : "";

  if (in_new.empty() || out_combined.empty() || var_token.empty() || tag.empty()) {
    throw std::runtime_error("merge_result_csv_two: missing required arguments (need in_new,out_combined,var_token,tag)");
  }
  if (gSystem->AccessPathName(in_new.c_str())) {
    throw std::runtime_error("Input new CSV does not exist: " + in_new);
  }

  CSVTable newT = read_csv(in_new);
  VarCols vc = varcols_for(var_token);

  size_t newTopo = require_col(newT, "true_Topology", in_new);
  size_t newTrue = require_col(newT, vc.tcol, in_new);
  size_t newPred = require_col(newT, vc.pcol, in_new);

  // Extract base prefix from training-name header field in the NEW file
  std::string new_train_field = find_training_header_field(newT.header);
  std::string new_base_prefix = base_prefix_up_to_NpNpi(new_train_field);

  if (new_base_prefix.empty()) {
    std::string msg = "WARNING: Could not extract base prefix up to 'NpNpi' from header field '" + new_train_field +
                      "' in file " + in_new;
    std::cerr << msg << std::endl;
    append_warning(warnings_file, msg);
  }

  const size_t nrows = newT.rows.size();
  const bool bootstrap = (in_combined.empty() || gSystem->AccessPathName(in_combined.c_str()));

  if (bootstrap) {
    // Create fresh combined: true_Topology + var true/pred + LABEL(last)
    std::string label_header;
    if (!new_base_prefix.empty()) {
      label_header = new_base_prefix + "_single_var_" + tag + "_Topology";
    } else {
      label_header = "combo_label"; // fallback
    }

    std::vector<std::string> header;
    header.push_back("true_Topology");
    header.push_back(vc.tcol);
    header.push_back(vc.pcol);
    header.push_back(label_header); // LAST

    std::vector<std::vector<std::string>> rows;
    rows.reserve(nrows);
    for (size_t i=0;i<nrows;i++) {
      std::vector<std::string> r;
      r.reserve(4);
      r.push_back(newT.rows[i][newTopo]);
      r.push_back(newT.rows[i][newTrue]);
      r.push_back(newT.rows[i][newPred]);
      r.push_back("-999999");
      rows.push_back(std::move(r));
    }
    write_csv(out_combined, header, rows);
    return;
  }

  // Append to existing combined
  CSVTable comb = read_csv(in_combined);

  // Identify existing label column as the LAST column (we enforce this invariant)
  if (comb.header.size() < 2) {
    throw std::runtime_error("Combined CSV has too few columns: " + in_combined);
  }
  const size_t label_idx = comb.header.size() - 1;
  const std::string current_label_header = comb.header[label_idx];

  // Determine base prefix from the combined label header (if it contains "NpNpi" prefix form)
  // We stored it as "<base>_single_var_..._Topology", so take substring before "_single_var_".
  std::string comb_base_prefix;
  {
    auto pos = current_label_header.find("_single_var_");
    if (pos != std::string::npos) comb_base_prefix = current_label_header.substr(0, pos);
  }

  // If both prefixes exist and differ => warn
  if (!comb_base_prefix.empty() && !new_base_prefix.empty() && comb_base_prefix != new_base_prefix) {
    std::ostringstream oss;
    oss << "WARNING: Base prefix mismatch while appending tag=" << tag << "\n"
        << "  Combined prefix: " << comb_base_prefix << "\n"
        << "  New-file prefix: " << new_base_prefix << "\n"
        << "  Combined file:   " << in_combined << "\n"
        << "  New file:        " << in_new;
    std::string msg = oss.str();
    std::cerr << msg << std::endl;
    append_warning(warnings_file, msg);
    // Continue (per your request: warn, do not necessarily abort)
  }

  // Check topology exists & matches
  size_t combTopo = require_col(comb, "true_Topology", in_combined);
  if (comb.rows.size() != nrows) {
    throw std::runtime_error("Row count mismatch between combined (" + std::to_string(comb.rows.size()) +
                             ") and new (" + std::to_string(nrows) + ")");
  }
  for (size_t i=0;i<nrows;i++) {
    long double a = parse_ld(comb.rows[i][combTopo]);
    long double b = parse_ld(newT.rows[i][newTopo]);
    if (!approx_equal(a,b)) {
      std::ostringstream oss;
      oss << "ERROR: true_Topology mismatch at row " << i
          << "\n  combined=" << (double)a
          << "\n  new=" << (double)b
          << "\nAborting.";
      throw std::runtime_error(oss.str());
    }
  }

  // Ensure we do NOT already have these columns
  if (comb.idx.count(vc.tcol) || comb.idx.count(vc.pcol)) {
    throw std::runtime_error("Combined already contains columns for " + var_token +
                             " (" + vc.tcol + " / " + vc.pcol + ")");
  }

  // --- Maintain invariant: label column is LAST ---
  // Temporarily remove the label column from header & each row
  comb.header.pop_back();
  for (auto& r : comb.rows) {
    if (r.size() != comb.header.size() + 1) {
      throw std::runtime_error("Internal error: row/header size mismatch in combined before relabel");
    }
    r.pop_back();
  }

  // Append new columns
  comb.header.push_back(vc.tcol);
  comb.header.push_back(vc.pcol);
  for (size_t i=0;i<nrows;i++) {
    comb.rows[i].push_back(newT.rows[i][newTrue]);
    comb.rows[i].push_back(newT.rows[i][newPred]);
  }

  // Update label header to include this tag; re-add label column as LAST
  std::string new_label_header = update_label_header_append(current_label_header, tag);
  comb.header.push_back(new_label_header);
  for (auto& r : comb.rows) r.push_back("-999999");

  // Rebuild idx
  comb.idx.clear();
  for (size_t i=0;i<comb.header.size();i++) comb.idx[comb.header[i]] = i;

  write_csv(out_combined, comb.header, comb.rows);
}
