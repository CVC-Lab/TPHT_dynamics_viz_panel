#include "tpht/native_controller.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

namespace tpht {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kNv = 9, kNu = 6;

double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

}  // namespace

NativeMethod parse_native_method(const std::string& id) {
  if (id == "mujoco_ls") return NativeMethod::ls;
  if (id == "mujoco_fd_ls") return NativeMethod::fd_ls;
  if (id == "mujoco_mpc") return NativeMethod::mpc;
  throw BackendError("method", "Unknown native method " + id);
}

const char* native_method_id(NativeMethod m) {
  switch (m) {
    case NativeMethod::ls: return "mujoco_ls";
    case NativeMethod::fd_ls: return "mujoco_fd_ls";
    case NativeMethod::mpc: return "mujoco_mpc";
  }
  return "?";
}

NativeController::NativeController(Sim& sim, const GaitConfig& cfg, NativeMethod method, uint32_t seed,
                                   double fd_epsilon, double trust_radius)
    : sim_(sim),
      cfg_(cfg),
      method_(method),
      fd_epsilon_(fd_epsilon),
      trust_radius_(trust_radius),
      last_plan_(-std::numeric_limits<double>::infinity()),
      first_hold_(cfg.options.planning_dt),
      randn_(seed) {
  if (sim.nv() != kNv || sim.nu() != kNu) {
    throw BackendError("model-shape", "The native controllers are written for the planar cheetah (nv=9, nu=6); got nv=" +
                                          std::to_string(sim.nv()) + " nu=" + std::to_string(sim.nu()));
  }
  for (int i = 0; i < kGaitParams; i++) mean_[i] = std::min(cfg.high[i], std::max(cfg.low[i], cfg.gait[i]));
}

void NativeController::forward(mjData* d, const double* action) {
  sim_.forward(d, action);
  counters_.forward_calls++;
}

std::array<double, kAccelRows> NativeController::desired(const mjData* d, const Scan& scan, const GaitParams& params,
                                                         double phase, double target) const {
  const TorsoState st = sim_.torso_state(d);
  TorsoInput in;
  in.qpos = st.qpos;
  in.qvel = st.qvel;
  std::copy(st.torso_pos, st.torso_pos + 3, in.torso_pos);
  in.torso_linvel_x = st.torso_linvel_x;
  return desired_acceleration(in, scan, params, phase, target, cfg_.options, cfg_.limits);
}

FitResult NativeController::fit(mjData* d, const Scan& scan, const GaitParams& params, double phase, double target,
                                bool finite_difference, bool diagnose) {
  forward(d);
  const auto want = desired(d, scan, params, phase, target);
  double center[kNu], base[kNv];
  std::copy(d->ctrl, d->ctrl + kNu, center);
  std::copy(d->qacc, d->qacc + kNv, base);

  // Acceleration map with the measured current constraint force frozen.
  const Vec M = sim_.full_mass(d);
  Vec J = solve_spd(M.data(), kNv, sim_.B().data(), kNu);
  double offset[kNv];
  {
    const Vec Ju = mat_vec(J.data(), kNv, kNu, d->ctrl);
    for (int i = 0; i < kNv; i++) offset[i] = d->qacc[i] - Ju[i];
  }
  double low[kNu], high[kNu];
  std::fill(low, low + kNu, -1.0);
  std::fill(high, high + kNu, 1.0);
  if (finite_difference) {
    J.assign(kNv * kNu, 0.0);
    mjData* dm = sim_.scratch(2);
    mjData* dp = sim_.scratch(3);
    for (int j = 0; j < kNu; j++) {
      double minus[kNu], plus[kNu];
      std::copy(center, center + kNu, minus);
      std::copy(center, center + kNu, plus);
      minus[j] = std::max(-1.0, center[j] - fd_epsilon_);
      plus[j] = std::min(1.0, center[j] + fd_epsilon_);
      sim_.copy_into(dm, d);
      sim_.copy_into(dp, d);
      forward(dm, minus);
      forward(dp, plus);
      for (int i = 0; i < kNv; i++) J[i * kNu + j] = (dp->qacc[i] - dm->qacc[i]) / (plus[j] - minus[j]);
    }
    const Vec Jc = mat_vec(J.data(), kNv, kNu, center);
    for (int i = 0; i < kNv; i++) offset[i] = base[i] - Jc[i];
    for (int j = 0; j < kNu; j++) {
      low[j] = std::max(-1.0, center[j] - trust_radius_);
      high[j] = std::min(1.0, center[j] + trust_radius_);
    }
  }
  const BoundedLsResult ls = bounded_ls(J.data(), offset, want.data(), cfg_.options.feedback_ridge, low, high);
  counters_.feedback_solves++;
  FitResult out;
  out.action = ls.action;
  out.ls = ls.diagnostics;
  if (diagnose) {
    mjData* actual = sim_.copy_into(sim_.scratch(4), d);
    forward(actual, out.action.data());
    const Vec lin0 = mat_vec(J.data(), kNv, kNu, out.action.data());
    double err[kNv];
    for (int i = 0; i < kNv; i++) {
      out.acceleration_linear_prediction[i] = lin0[i] + offset[i];
      out.acceleration_full_contact[i] = actual->qacc[i];
      err[i] = actual->qacc[i] - out.acceleration_linear_prediction[i];
    }
    out.acceleration_target = want;
    out.contact_linearization_error = norm(err, kNv);
    int saturated = 0;
    for (const double x : out.action) saturated += std::abs(x) > 0.99;
    out.action_saturation = saturated / 6.0;
    out.diagnosed = true;
  }
  return out;
}

void NativeController::rollout(const Scan& scan, const std::vector<GaitParams>& population, double target,
                               double phase, std::vector<double>& costs, std::vector<bool>& failures) {
  const ControllerOptions& o = cfg_.options;
  const mjModel* m = sim_.model();
  const double dt = m->opt.timestep;
  Vec xs(scan.size()), hs(scan.size());
  for (size_t k = 0; k < scan.size(); k++) { xs[k] = scan[k][0]; hs[k] = scan[k][1]; }
  const double* jr = m->jnt_range;
  costs.clear();
  failures.clear();
  mjData* d = sim_.scratch(1);
  for (const GaitParams& cand : population) {
    sim_.copy_into(d, sim_.data());
    forward(d);
    const double start = sim_.torso_state(d).torso_com_x;
    double elapsed = 0, angle = phase, cost = 0;
    bool invalid = false;
    for (int interval = 0; interval < o.horizon_steps && !invalid; interval++) {
      const double duration = interval == 0 ? first_hold_ : o.planning_dt;
      const int count = std::max(1, static_cast<int>(std::floor(duration / dt + 0.5)));  // Math.round
      const FitResult f = fit(d, scan, cand, angle, target);
      std::copy(f.action.begin(), f.action.end(), d->ctrl);
      mj_step1(m, d);  // restore the split-step invariant after mj_forward
      double effort = 0;
      for (const double x : f.action) effort += x * x;
      for (int k = 0; k < count; k++) {
        if (sim_.legacy_step()) { mj_step2(m, d); mj_step1(m, d); }
        else mj_step(m, d);
        mj_subtreeVel(m, d);
        counters_.native_steps++;
        const TorsoState st = sim_.torso_state(d);
        const double clearance = st.torso_pos[2] - interp(st.torso_pos[0], xs, hs);
        const double speed = st.torso_linvel_x;
        double stage = o.speed_weight * (speed - target) * (speed - target) +
                       o.clearance_weight * (clearance - o.target_clearance) * (clearance - o.target_clearance);
        stage += o.pitch_weight * d->qpos[2] * d->qpos[2] + o.effort_weight * effort;
        stage += o.uncertainty_weight * (st.torso_pos[0] < xs.front() || st.torso_pos[0] > xs.back() ? 1 : 0);
        cost += dt * stage;
        elapsed += dt;
        angle += 2 * kPi * cand[18] * dt;
        bool bad = clearance < 0.15 || std::abs(d->qpos[2]) > 1.6;
        for (int i = 0; i < kNv && !bad; i++) {
          if (!std::isfinite(d->qpos[i]) || !std::isfinite(d->qvel[i]) || std::abs(d->qvel[i]) > 80) bad = true;
        }
        for (int j = 3; j < kNv && !bad; j++) {
          if (d->qpos[j] < jr[j * 2] - 0.3 || d->qpos[j] > jr[j * 2 + 1] + 0.3) bad = true;
        }
        if (bad) { invalid = true; break; }
      }
    }
    const TorsoState st = sim_.torso_state(d);
    cost += o.progress_weight * std::max(0.0, target * elapsed - (st.torso_com_x - start)) +
            0.1 * (st.torso_linvel_x - target) * (st.torso_linvel_x - target);
    costs.push_back(cost + 1e6 * (invalid ? 1 : 0));
    failures.push_back(invalid);
  }
}

void NativeController::plan(const Scan& scan, double target) {
  const ControllerOptions& o = cfg_.options;
  GaitParams mean = mean_, std_dev = cfg_.scale, best = mean_;
  double best_cost = std::numeric_limits<double>::infinity(), fraction = 0;
  auto clamp_p = [&](GaitParams p) {
    for (int i = 0; i < kGaitParams; i++) p[i] = std::min(cfg_.high[i], std::max(cfg_.low[i], p[i]));
    return p;
  };
  std::vector<GaitParams> population(o.candidates);
  std::vector<double> costs;
  std::vector<bool> failures;
  std::vector<int> order;
  for (int it = 0; it < o.iterations; it++) {
    for (auto& cand : population) {
      GaitParams p;
      for (int i = 0; i < kGaitParams; i++) p[i] = mean[i] + randn_() * std_dev[i];
      cand = clamp_p(p);
    }
    if (!population.empty()) population[0] = best;
    rollout(scan, population, target, phase_, costs, failures);
    order.resize(costs.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return costs[a] < costs[b]; });
    order.resize(std::min<size_t>(order.size(), static_cast<size_t>(o.elites)));
    if (order.empty()) break;
    if (costs[order[0]] < best_cost) { best_cost = costs[order[0]]; best = population[order[0]]; }
    for (int i = 0; i < kGaitParams; i++) {
      double s = 0;
      for (const int k : order) s += population[k][i];
      const double e_mean = s / order.size();
      double v = 0;
      for (const int k : order) v += (population[k][i] - e_mean) * (population[k][i] - e_mean);
      const double e_std = std::sqrt(v / order.size());
      mean[i] = 0.25 * mean[i] + 0.75 * e_mean;
      std_dev[i] = std::max(0.015, 0.25 * std_dev[i] + 0.75 * e_std);
    }
    fraction = static_cast<double>(std::count(failures.begin(), failures.end(), true)) / failures.size();
  }
  mean_ = best;
  counters_.plans++;
  diag_.has_plan = true;
  diag_.planner_cost = best_cost;
  diag_.planner_invalid_fraction = fraction;
  diag_.frequency_hz = best[18];
}

std::array<float, kActions> NativeController::act(const Scan& scan, double target, double hold, double time_now) {
  const double started = now_ms();
  if (has_last_time_) {
    phase_ = std::fmod(phase_ + 2 * kPi * mean_[18] * std::max(0.0, time_now - last_time_), 2 * kPi);
  }
  has_last_time_ = true;
  last_time_ = time_now;
  first_hold_ = hold;
  bool planned = false;
  if (method_ == NativeMethod::mpc && time_now - last_plan_ >= cfg_.options.replan_seconds - 1e-9) {
    plan(scan, target);
    last_plan_ = time_now;
    planned = true;
  }
  mjData* d = sim_.copy_into(sim_.scratch(0), sim_.data());
  const FitResult f = fit(d, scan, mean_, phase_, target, method_ == NativeMethod::fd_ls, true);
  diag_.fit = f;
  diag_.phase = phase_;
  diag_.planned = planned;
  diag_.decision_ms = now_ms() - started;
  // The recorded environment used float32 controls.
  std::array<float, kActions> out{};
  for (int i = 0; i < kActions; i++) out[i] = static_cast<float>(f.action[i]);
  return out;
}

}  // namespace tpht
