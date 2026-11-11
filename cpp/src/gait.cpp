#include "tpht/gait.h"

#include <algorithm>
#include <cmath>

#include <nlohmann/json.hpp>

namespace tpht {

namespace {

constexpr double kPi = 3.14159265358979323846;

template <size_t N>
std::array<double, N> fixed(const nlohmann::json& j, const char* what) {
  if (!j.is_array() || j.size() != N) {
    throw std::runtime_error(std::string("gait.json: ") + what + " must have " + std::to_string(N) + " numbers");
  }
  std::array<double, N> out{};
  for (size_t i = 0; i < N; i++) out[i] = j[i].get<double>();
  return out;
}

template <typename T>
void opt(const nlohmann::json& j, const char* key, T& out) {
  if (auto it = j.find(key); it != j.end() && !it->is_null()) out = it->get<T>();
}

}  // namespace

ControllerOptions ControllerOptions::from_json(const nlohmann::json& j) {
  ControllerOptions o;
  // The feedback law needs these; the exporter always writes them.
  o.speed_kd = j.at("speed_kd").get<double>();
  o.height_kp = j.at("height_kp").get<double>();
  o.height_kd = j.at("height_kd").get<double>();
  o.pitch_kp = j.at("pitch_kp").get<double>();
  o.pitch_kd = j.at("pitch_kd").get<double>();
  o.joint_kp = j.at("joint_kp").get<double>();
  o.joint_kd = j.at("joint_kd").get<double>();
  o.target_clearance = j.at("target_clearance").get<double>();
  opt(j, "feedback_ridge", o.feedback_ridge);
  opt(j, "planning_dt", o.planning_dt);
  opt(j, "replan_seconds", o.replan_seconds);
  opt(j, "horizon_steps", o.horizon_steps);
  opt(j, "iterations", o.iterations);
  opt(j, "candidates", o.candidates);
  opt(j, "elites", o.elites);
  opt(j, "speed_weight", o.speed_weight);
  opt(j, "clearance_weight", o.clearance_weight);
  opt(j, "pitch_weight", o.pitch_weight);
  opt(j, "effort_weight", o.effort_weight);
  opt(j, "uncertainty_weight", o.uncertainty_weight);
  opt(j, "progress_weight", o.progress_weight);
  return o;
}

GaitConfig GaitConfig::from_json(const nlohmann::json& j) {
  GaitConfig g;
  g.gait = fixed<kGaitParams>(j.at("gait"), "gait");
  g.low = fixed<kGaitParams>(j.at("low"), "low");
  g.high = fixed<kGaitParams>(j.at("high"), "high");
  g.scale = fixed<kGaitParams>(j.at("scale"), "scale");
  const auto& lim = j.at("limits");
  if (!lim.is_array() || lim.size() != 6) throw std::runtime_error("gait.json: limits must be 6 [low, high] pairs");
  for (size_t i = 0; i < 6; i++) g.limits[i] = fixed<2>(lim[i], "limits[i]");
  g.options = ControllerOptions::from_json(j.at("options"));
  return g;
}

References references(const GaitParams& params, double phase, const JointLimits& limits) {
  References r;
  const double omega = 2 * kPi * params[18];
  const double s = std::sin(phase), c = std::cos(phase);
  for (int i = 0; i < 6; i++) {
    const double wave = params[6 + i] * s + params[12 + i] * c;
    const double qi = params[i] + wave;
    const double lo = limits[i][0] + 0.02, hi = limits[i][1] - 0.02;
    const bool limited = qi > hi || qi < lo;  // evaluated before clamping, as in Python
    r.q[i] = std::min(hi, std::max(lo, qi));
    r.v[i] = limited ? 0 : omega * (params[6 + i] * c - params[12 + i] * s);
    r.a[i] = limited ? 0 : -omega * omega * wave;
  }
  return r;
}

std::array<double, 9> desired_acceleration(const TorsoInput& st, const Scan& scan, const GaitParams& params,
                                           double phase, double target, const ControllerOptions& o,
                                           const JointLimits& limits) {
  const References ref = references(params, phase, limits);
  Vec xs(scan.size()), hs(scan.size());
  for (size_t k = 0; k < scan.size(); k++) { xs[k] = scan[k][0]; hs[k] = scan[k][1]; }
  const double height = interp(st.torso_pos[0], xs, hs);
  std::array<double, 9> out{};
  out[0] = o.speed_kd * (target - st.torso_linvel_x);
  out[1] = o.height_kp * (o.target_clearance - (st.torso_pos[2] - height)) - o.height_kd * st.qvel[1];
  out[2] = -o.pitch_kp * st.qpos[2] - o.pitch_kd * st.qvel[2];
  for (int i = 0; i < 6; i++) {
    out[3 + i] = ref.a[i] + o.joint_kp * (ref.q[i] - st.qpos[3 + i]) + o.joint_kd * (ref.v[i] - st.qvel[3 + i]);
  }
  return out;
}

Profile Profile::from_json(const nlohmann::json& j) {
  Profile p;
  if (j.is_object() && j.contains("x") && j.contains("z")) {
    p.x = j.at("x").get<Vec>();
    p.z = j.at("z").get<Vec>();
    if (p.x.size() != p.z.size() || p.x.empty()) throw std::runtime_error("profile: x and z must be non-empty and the same length");
  }
  return p;
}

const std::array<double, 7> kScanOffsets = {-1, -0.5, 0, 0.5, 1, 2, 3};

Scan privileged_scan(double root_x, const Profile& profile) {
  Scan out;
  out.reserve(kScanOffsets.size());
  for (const double dx : kScanOffsets) {
    const double x = root_x + dx;
    const double h = interp(x, profile.x, profile.z);
    const double e = 0.05;
    const double slope = (interp(x + e, profile.x, profile.z) - interp(x - e, profile.x, profile.z)) / (2 * e);
    out.push_back({x, h, slope});
  }
  return out;
}

}  // namespace tpht
