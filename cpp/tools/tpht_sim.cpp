// tpht-sim: the C++ backend natively, on the same assets the page loads.
//
//   tpht-sim check  [--assets DIR]                         binding self-tests per model
//   tpht-sim run    [--assets DIR] --section ctrl [--policy demo-sine] [--seconds 12]
//   tpht-sim run    [--assets DIR] --section reference --case F --method mujoco_ls [--initial sample]
//   tpht-sim golden [--assets DIR] --case F [--clock fixed|irregular]
//   tpht-sim version
//
// Every command takes --json for machine-readable output. DIR defaults to
// public/assets when run from the repository, else <prefix>/share/tpht-viz-panel/assets.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tpht/backend.h"

namespace fs = std::filesystem;
using nlohmann::json;
using tpht::Backend;

namespace {

constexpr const char* kVersion = "0.1.0";

struct Args {
  std::string cmd;
  std::map<std::string, std::string> opt;
  bool flag(const std::string& k) const { return opt.count(k) != 0; }
  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = opt.find(k);
    return it == opt.end() ? def : it->second;
  }
};

Args parse_args(int argc, char** argv) {
  Args a;
  if (argc > 1) a.cmd = argv[1];
  for (int i = 2; i < argc; i++) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument " + k);
    k = k.substr(2);
    if (k == "json" || k == "help") { a.opt[k] = "1"; continue; }
    if (i + 1 >= argc) throw std::runtime_error("--" + k + " needs a value");
    a.opt[k] = argv[++i];
  }
  return a;
}

std::string read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw tpht::BackendError("asset-missing", "Missing " + p.string());
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

json read_json(const fs::path& p) { return json::parse(read_file(p)); }

fs::path default_assets(const char* argv0) {
  if (fs::is_directory("public/assets")) return "public/assets";
  std::error_code ec;
  fs::path exe = fs::weakly_canonical(fs::path(argv0), ec);
  fs::path installed = exe.parent_path().parent_path() / "share" / "tpht-viz-panel" / "assets";
  if (fs::is_directory(installed)) return installed;
  return "public/assets";
}

// A section's model: { asset, meta JSON text, expect }.
struct Loaded {
  tpht::ModelAsset asset;
  std::string meta;
  tpht::SelfTestExpect expect;
  json entry;  // the reference case, or the ctrl manifest
};

Loaded locate(const fs::path& assets, const std::string& section, const std::string& case_id) {
  Loaded L;
  const json manifest = read_json(assets / section / "manifest.json");
  json spec, meta_path;
  if (section == "reference") {
    if (!manifest.at("cases").contains(case_id)) {
      throw tpht::BackendError("case-missing", "Case " + case_id + " is not in reference/manifest.json");
    }
    L.entry = manifest["cases"][case_id];
    L.entry["target_speed"] = manifest.value("target_speed", 3.0);
    spec = L.entry.at("model");
    meta_path = L.entry.value("meta", json());
  } else {
    L.entry = manifest;
    spec = manifest.at("model");
    meta_path = manifest.value("meta", json());
  }
  const fs::path file = assets / section / spec.at("file").get<std::string>();
  L.asset.kind = spec.at("kind").get<std::string>();
  L.asset.name = file.filename().string();
  L.asset.data = read_file(file);
  if (meta_path.is_string()) {
    const fs::path mp = assets / section / meta_path.get<std::string>();
    if (fs::exists(mp)) L.meta = read_file(mp);
  }
  const json meta = L.meta.empty() ? json::object() : json::parse(L.meta);
  if (meta.contains("timestep")) L.expect.timestep = meta["timestep"].get<double>();
  if (section == "reference") {
    L.expect.dims = std::array<int, 3>{9, 9, 6};
    L.expect.euler = true;
  }
  return L;
}

void print_tests(const std::string& label, const std::vector<tpht::SelfTest>& tests, int& failed) {
  for (const auto& t : tests) {
    std::printf("%s  [%s] %s: %s\n", t.ok ? "PASS" : "FAIL", label.c_str(), t.name.c_str(), t.detail.c_str());
    failed += !t.ok;
  }
}

int cmd_check(const Args& a, const fs::path& assets) {
  int failed = 0;
  json report = json::object();
  std::printf("tpht-sim: MuJoCo %s, assets %s\n", Backend::mujoco_version().c_str(), assets.string().c_str());
  auto run = [&](const std::string& section, const std::string& case_id, const std::string& label) {
    Backend b;
    try {
      Loaded L = locate(assets, section, case_id);
      const auto r = b.load(L.asset, L.meta, L.expect);
      report[label] = tpht::to_json(r.self_tests);
      if (!a.flag("json")) print_tests(label, r.self_tests, failed);
      else for (const auto& t : r.self_tests) failed += !t.ok;
    } catch (const tpht::BackendError& e) {
      // A git-ignored reference model is expected to be missing.
      const bool skip = e.code() == "asset-missing" && section == "reference";
      report[label] = {{{"name", "Model loader"}, {"ok", skip}, {"detail", (skip ? "skipped: " : "") + std::string(e.what())}}};
      if (!a.flag("json")) std::printf("%s  [%s] Model loader%s: %s\n", skip ? "PASS" : "FAIL", label.c_str(), skip ? " (skipped)" : "", e.what());
      failed += !skip;
    }
  };
  run("ctrl", "", "ctrl");
  if (fs::exists(assets / "reference" / "manifest.json")) {
    const json manifest = read_json(assets / "reference" / "manifest.json");
    for (auto it = manifest.at("cases").begin(); it != manifest["cases"].end(); ++it) {
      run("reference", it.key(), "reference/" + it.key());
    }
  }
  if (a.flag("json")) std::cout << json({{"mujoco", Backend::mujoco_version()}, {"results", report}, {"failed", failed}}).dump(2) << "\n";
  else std::printf("%d failed\n", failed);
  return failed ? 1 : 0;
}

int cmd_run(const Args& a, const fs::path& assets) {
  const std::string section = a.get("section", "ctrl");
  const std::string case_id = a.get("case", "F");
  const double seconds = std::stod(a.get("seconds", "12"));
  Backend b;
  Loaded L = locate(assets, section, case_id);
  b.load(L.asset, L.meta, L.expect);
  double target = std::stod(a.get("target", "3"));
  std::string method;
  if (section == "reference") {
    target = std::stod(a.get("target", std::to_string(L.entry.value("target_speed", 3.0))));
    method = a.get("method", "mujoco_ls");
    const fs::path gp = assets / "reference" / L.entry.at("gait").get<std::string>();
    b.set_gait(read_file(gp));
    // The sample's pipeline input is the initial state (the page's "sample" start).
    const json sample = read_json(assets / "reference" / L.entry.at("sample").get<std::string>());
    const auto& p = sample.at("pipeline_input");
    const auto q = p.at("state_backend_q").get<std::vector<double>>();
    const auto v = p.at("state_backend_v").get<std::vector<double>>();
    const auto u = p.at("previous_action").get<std::vector<double>>();
    b.set_state(q.data(), v.data(), u.data());
    b.use_native(method, static_cast<uint32_t>(std::stoul(a.get("seed", "100000"))));
  } else {
    method = a.get("policy", tpht::SineGait::kId);
    if (method != tpht::SineGait::kId) {
      throw tpht::BackendError("policy-native", "tpht-sim runs the built-in " + std::string(tpht::SineGait::kId) +
                                                    " gait; exported ONNX policies run in the page for now.");
    }
    b.reset_data();
    b.use_sine_gait();
  }
  tpht::Sim& sim = b.sim();
  const double dt = sim.model()->opt.timestep;
  const double hold = std::stod(a.get("hold", "0.01"));
  const int substeps = std::max(1, static_cast<int>(std::floor(hold / dt + 0.5)));
  const double x0 = sim.data()->qpos[0];
  int n = 0;
  double time = 0, sq_err = 0, sat = 0, dec_ms = 0, max_pitch = 0, reward = 0;
  long plans = 0;
  bool finite = true;
  while (sim.data()->time < seconds - 1e-9) {
    const auto action = b.decide(target, substeps * dt);
    const json diag = b.diagnostics();
    b.step_hold(action.data(), static_cast<int>(action.size()), substeps);
    const auto st = sim.torso_state(sim.data());
    const double h = substeps * dt;
    n++;
    time += h;
    sq_err += h * (st.torso_linvel_x - target) * (st.torso_linvel_x - target);
    sat += h * diag.value("action_saturation", 0.0);
    dec_ms += diag.value("decision_ms", 0.0);
    reward += h * std::min(1.0, std::max(0.0, st.torso_linvel_x / 10));  // dm_control cheetah-run reward rate
    plans += diag.value("planned", false) ? 1 : 0;
    max_pitch = std::max(max_pitch, std::abs(sim.data()->qpos[2]));
    for (int i = 0; i < sim.model()->nq; i++) finite = finite && std::isfinite(sim.data()->qpos[i]);
    if (!finite) break;
  }
  const json out = {
      {"section", section}, {"method", method}, {"mujoco", Backend::mujoco_version()},
      {"seconds", sim.data()->time}, {"decisions", n}, {"distance", sim.data()->qpos[0] - x0},
      {"speed_rmse", std::sqrt(sq_err / std::max(time, 1e-12))}, {"saturation", sat / std::max(time, 1e-12)},
      {"decision_ms", dec_ms / std::max(n, 1)}, {"reward", reward}, {"plans", plans},
      {"max_pitch", max_pitch}, {"finite", finite}};
  if (a.flag("json")) {
    std::cout << out.dump(2) << "\n";
  } else {
    std::printf("tpht-sim run: %s %s, %.2f s, %d decisions: moved %+.3f m, max pitch %.3f rad, speed RMSE %.3f m/s, "
                "%.3f ms per decision, %ld plans, state finite %s\n",
                section.c_str(), method.c_str(), sim.data()->time, n, sim.data()->qpos[0] - x0, max_pitch,
                out["speed_rmse"].get<double>(), out["decision_ms"].get<double>(), plans, finite ? "yes" : "no");
  }
  return finite ? 0 : 1;
}

int cmd_golden(const Args& a, const fs::path& assets) {
  const std::string case_id = a.get("case", "F");
  const std::string clock = a.get("clock", "fixed");
  Backend b;
  Loaded L = locate(assets, "reference", case_id);
  b.load(L.asset, L.meta, L.expect);
  if (L.entry.contains("gait") && fs::exists(assets / "reference" / L.entry["gait"].get<std::string>())) {
    b.set_gait(read_file(assets / "reference" / L.entry["gait"].get<std::string>()));
  }
  std::string sample = L.entry.at("sample").get<std::string>();
  const auto slash = sample.find('/', sample.find('/') + 1);
  sample = "samples/" + clock + sample.substr(slash);
  const auto checks = b.golden(read_file(assets / "reference" / sample));
  int failed = 0;
  for (const auto& c : checks) failed += c.value && c.tol && !(*c.value <= *c.tol);
  if (a.flag("json")) {
    std::cout << json({{"case", case_id}, {"clock", clock}, {"results", tpht::to_json(checks)}, {"failed", failed}}).dump(2) << "\n";
  } else {
    for (const auto& c : checks) {
      if (!c.value) { std::printf("SKIP  %s: %s\n", c.name.c_str(), c.note.c_str()); continue; }
      std::printf("%s  %s: %.3e (tol %.0e)\n", *c.value <= *c.tol ? "PASS" : "FAIL", c.name.c_str(), *c.value, *c.tol);
    }
  }
  return failed ? 1 : 0;
}

void usage() {
  std::fputs(
      "usage: tpht-sim <check|run|golden|version> [--assets DIR] [--json]\n"
      "  check                                   binding self-tests on every model in DIR\n"
      "  run --section ctrl [--policy demo-sine] [--seconds 12] [--hold 0.01]\n"
      "  run --section reference --case F --method mujoco_ls|mujoco_fd_ls|mujoco_mpc [--seconds 3] [--seed 100000]\n"
      "  golden --case F [--clock fixed|irregular]\n",
      stderr);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = parse_args(argc, argv);
    if (a.cmd.empty() || a.cmd == "help" || a.flag("help")) { usage(); return a.cmd.empty() ? 2 : 0; }
    if (a.cmd == "version" || a.cmd == "--version") {
      std::printf("tpht-sim %s (MuJoCo %s)\n", kVersion, Backend::mujoco_version().c_str());
      return 0;
    }
    const fs::path assets = a.flag("assets") ? fs::path(a.get("assets")) : default_assets(argv[0]);
    if (a.cmd == "check") return cmd_check(a, assets);
    if (a.cmd == "run") return cmd_run(a, assets);
    if (a.cmd == "golden") return cmd_golden(a, assets);
    usage();
    return 2;
  } catch (const tpht::BackendError& e) {
    std::fprintf(stderr, "tpht-sim: %s (%s)\n", e.what(), e.code().c_str());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tpht-sim: %s\n", e.what());
    return 1;
  }
}
