#include "tpht/backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace tpht {

namespace {

constexpr double kPi = 3.14159265358979323846;

double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

nlohmann::json parse_or_empty(const std::string& text, const char* what) {
  if (text.empty()) return nlohmann::json::object();
  try {
    return nlohmann::json::parse(text);
  } catch (const std::exception& e) {
    throw BackendError("bad-json", std::string(what) + " is not valid JSON: " + e.what());
  }
}

template <size_t N>
double max_abs_diff_json(const std::array<double, N>& a, const nlohmann::json& b) {
  double s = 0;
  for (size_t i = 0; i < N && i < b.size(); i++) s = std::max(s, std::abs(a[i] - b[i].get<double>()));
  return s;
}

}  // namespace

SineGait::SineGait(int nu, double frequency, double amplitude, double joint_lag, double leg_offset)
    : frequency_(frequency), amplitude_(amplitude), phase_(nu) {
  // Phase per actuator: thigh, shin, foot each joint_lag behind the last; front leg offset by leg_offset.
  for (int i = 0; i < nu; i++) phase_[i] = (i % 3) * joint_lag + (i >= 3 ? leg_offset : 0);
}

std::vector<float> SineGait::act(double time) const {
  const double w = 2 * kPi * frequency_ * time;
  std::vector<float> action(phase_.size());
  for (size_t i = 0; i < phase_.size(); i++) action[i] = static_cast<float>(amplitude_ * std::sin(w + phase_[i]));
  return action;
}

std::string Backend::mujoco_version() { return mj_versionString(); }

Sim& Backend::sim() {
  if (!sim_) throw BackendError("not-ready", "Model is not loaded yet.");
  return *sim_;
}

LoadResult Backend::load(const ModelAsset& asset, const std::string& meta_json, const SelfTestExpect& expect) {
  clear_controller();
  sim_.reset();
  meta_ = parse_or_empty(meta_json, "Model metadata");
  profile_ = Profile::from_json(meta_.value("profile", nlohmann::json()));
  LoadResult out;
  ModelPtr model = load_model(asset, &out.via);
  sim_ = std::make_unique<Sim>(std::move(model), meta_);
  out.self_tests.push_back({"Model loader", true, out.via});
  auto tests = sim_->self_test(expect);
  out.self_tests.insert(out.self_tests.end(), tests.begin(), tests.end());
  return out;
}

void Backend::set_gait(const std::string& gait_json) {
  native_.reset();
  if (gait_json.empty()) {
    gait_.reset();
    return;
  }
  try {
    gait_ = GaitConfig::from_json(nlohmann::json::parse(gait_json));
  } catch (const BackendError&) {
    throw;
  } catch (const std::exception& e) {
    throw BackendError("gait-invalid", std::string("gait.json: ") + e.what());
  }
}

void Backend::reset_data() { sim().reset_data(); }

void Backend::set_state(const double* q, const double* v, const double* ctrl) { sim().set_state(q, v, ctrl); }

void Backend::use_native(const std::string& method, uint32_t seed) {
  if (!gait_) {
    throw BackendError("gait-missing",
                       "No gait.json for this case. Native controllers need the saved gait from final.pt "
                       "(export_web_assets.py gait).");
  }
  sine_.reset();
  native_ = std::make_unique<NativeController>(sim(), *gait_, parse_native_method(method), seed);
}

void Backend::use_sine_gait() {
  native_.reset();
  sine_ = std::make_unique<SineGait>(sim().nu());
}

void Backend::clear_controller() {
  native_.reset();
  sine_.reset();
}

std::vector<float> Backend::decide(double target, double hold) {
  Sim& s = sim();
  if (native_) {
    const Scan scan = privileged_scan(s.data()->qpos[0], profile_);
    const auto a = native_->act(scan, target, hold, s.data()->time);
    return std::vector<float>(a.begin(), a.end());
  }
  if (sine_) {
    const double started = now_ms();
    auto a = sine_->act(s.data()->time);
    sine_decision_ms_ = now_ms() - started;
    return a;
  }
  throw BackendError("no-controller", "No controller is selected; reset first.");
}

nlohmann::json Backend::diagnostics() const {
  nlohmann::json j = nlohmann::json::object();
  if (native_) {
    const ControllerDiagnostics& d = native_->diagnostics();
    if (d.fit.diagnosed) {
      j["ls_projected_gradient_inf"] = d.fit.ls.projected_gradient_inf;
      j["weighted_acceleration_residual"] = d.fit.ls.weighted_residual;
      j["ls_iterations"] = d.fit.ls.iterations;
      j["contact_linearization_error"] = d.fit.contact_linearization_error;
      j["action_saturation"] = d.fit.action_saturation;
    }
    if (d.has_plan) {
      j["planner_cost"] = d.planner_cost;
      j["planner_invalid_fraction"] = d.planner_invalid_fraction;
      j["frequency_hz"] = d.frequency_hz;
    }
    j["phase"] = d.phase;
    j["planned"] = d.planned;
    j["decision_ms"] = d.decision_ms;
  } else if (sine_) {
    j["decision_ms"] = sine_decision_ms_;
    j["action_saturation"] = 0;
  }
  return j;
}

void Backend::step_hold(const float* action, int n, int substeps) {
  Sim& s = sim();
  if (n != s.nu()) throw BackendError("action-size", "Expected " + std::to_string(s.nu()) + " controls, got " + std::to_string(n));
  std::vector<double> a(action, action + n);
  s.step_hold(a.data(), substeps);
}

std::vector<GoldenCheck> Backend::golden(const std::string& sample_json) {
  Sim& s = sim();
  const nlohmann::json sample = parse_or_empty(sample_json, "Reference sample");
  const auto& p = sample.at("pipeline_input");
  const auto& dec = sample.at("decision");
  const auto& rec = sample.at("received_transition");
  std::vector<GoldenCheck> out;
  const auto q0 = p.at("state_backend_q").get<std::vector<double>>();
  const auto v0 = p.at("state_backend_v").get<std::vector<double>>();
  const auto u0 = p.at("previous_action").get<std::vector<double>>();
  s.set_state(q0.data(), v0.data(), u0.data());
  const auto action = dec.at("action_normalized").get<std::vector<double>>();
  s.step_hold(action.data(), rec.at("physics_substeps").get<int>());
  const auto rq = rec.at("q").get<std::vector<double>>();
  const auto rv = rec.at("v").get<std::vector<double>>();
  out.push_back({"Transition replay q (max abs)", max_abs_diff(s.data()->qpos, rq.data(), s.model()->nq), 1e-6, ""});
  out.push_back({"Transition replay v (max abs)", max_abs_diff(s.data()->qvel, rv.data(), s.model()->nv), 1e-4, ""});

  if (gait_) {
    s.set_state(q0.data(), v0.data(), u0.data());
    const std::string method = dec.at("method").get<std::string>();
    NativeController ctl(s, *gait_, method == "mujoco_mpc" ? NativeMethod::ls : parse_native_method(method));
    mjData* d = s.copy_into(s.scratch(0), s.data());
    Scan scan;
    for (const auto& row : p.at("scan")) scan.push_back({row[0].get<double>(), row[1].get<double>(), row[2].get<double>()});
    const auto& dd = dec.at("diagnostics");
    const FitResult fit = ctl.fit(d, scan, ctl.mean(), dd.at("phase").get<double>(),
                                  p.at("target_speed_m_per_s").get<double>(), method == "mujoco_fd_ls", true);
    out.push_back({"Acceleration target (max abs)", max_abs_diff_json(fit.acceleration_target, dd.at("acceleration_target")), 1e-5, ""});
    out.push_back({"Fitted action, " + method + " (max abs)", max_abs_diff_json(fit.action, dec.at("action_normalized")), 1e-4, ""});
    out.push_back({"Full-contact acceleration (max abs)",
                   max_abs_diff_json(fit.acceleration_full_contact, dd.at("acceleration_full_contact")), 1e-3, ""});
  } else {
    out.push_back({"Controller checks", std::nullopt, std::nullopt, "Skipped: export gait.json from final.pt first."});
  }
  mj_resetData(s.model(), s.data());
  return out;
}

nlohmann::json to_json(const std::vector<SelfTest>& tests) {
  nlohmann::json j = nlohmann::json::array();
  for (const auto& t : tests) j.push_back({{"name", t.name}, {"ok", t.ok}, {"detail", t.detail}});
  return j;
}

nlohmann::json to_json(const std::vector<GoldenCheck>& checks) {
  nlohmann::json j = nlohmann::json::array();
  for (const auto& c : checks) {
    nlohmann::json r = {{"name", c.name}};
    r["value"] = c.value ? nlohmann::json(*c.value) : nlohmann::json(nullptr);
    r["tol"] = c.tol ? nlohmann::json(*c.tol) : nlohmann::json(nullptr);
    if (!c.note.empty()) r["note"] = c.note;
    j.push_back(r);
  }
  return j;
}

}  // namespace tpht
