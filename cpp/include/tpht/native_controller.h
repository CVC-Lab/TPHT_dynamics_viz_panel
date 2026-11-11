// The reference controllers that read MuJoCo itself (mujoco_ls, mujoco_fd_ls,
// mujoco_mpc). Port of src/controllers/native.js, which ports
// mujoco_reference/controller.py::NativeController plus the CEM loop of
// pure_cpht/controller.py::Controller.plan.
//
// Faithful parts: target construction, acceleration map, finite-difference
// Jacobian with trust box, bounded LS, rollout cost and validity rules, CEM
// update. The CEM noise uses the JavaScript backend's seeded generator
// (tpht/rng.h), not torch.Generator, so mujoco_mpc matches the Python arm in
// distribution only, and the JavaScript backend draw for draw.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "tpht/bvls.h"
#include "tpht/gait.h"
#include "tpht/rng.h"
#include "tpht/sim.h"

namespace tpht {

enum class NativeMethod { ls, fd_ls, mpc };
NativeMethod parse_native_method(const std::string& id);  // mujoco_ls, mujoco_fd_ls, mujoco_mpc
const char* native_method_id(NativeMethod m);

struct FitResult {
  std::array<double, kActions> action{};
  LsDiagnostics ls;
  bool diagnosed = false;
  std::array<double, kAccelRows> acceleration_target{}, acceleration_linear_prediction{},
      acceleration_full_contact{};
  double contact_linearization_error = 0;
  double action_saturation = 0;
};

// What the page shows per decision; planner fields persist from the last plan,
// as Object.assign(this.diagnostics, ...) does in native.js.
struct ControllerDiagnostics {
  FitResult fit;
  bool has_plan = false;
  double planner_cost = 0, planner_invalid_fraction = 0, frequency_hz = 0;
  double phase = 0;
  bool planned = false;
  double decision_ms = 0;
};

struct ControllerCounters {
  long long native_steps = 0, forward_calls = 0, feedback_solves = 0, plans = 0;
};

class NativeController {
 public:
  NativeController(Sim& sim, const GaitConfig& cfg, NativeMethod method, uint32_t seed = 100000,
                   double fd_epsilon = 1e-3, double trust_radius = 0.5);

  // The feedback solve on state d (which it runs mj_forward on).
  FitResult fit(mjData* d, const Scan& scan, const GaitParams& params, double phase, double target,
                bool finite_difference = false, bool diagnose = false);
  // One CEM replan of the gait parameters (mujoco_mpc).
  void plan(const Scan& scan, double target);
  // Mirrors NativeController.act: the action for the current state, rounded to
  // float32 like the recorded environment's controls.
  std::array<float, kActions> act(const Scan& scan, double target, double hold, double time_now);

  const ControllerDiagnostics& diagnostics() const { return diag_; }
  const ControllerCounters& counters() const { return counters_; }
  const GaitParams& mean() const { return mean_; }
  NativeMethod method() const { return method_; }

 private:
  void forward(mjData* d, const double* action = nullptr);
  std::array<double, kAccelRows> desired(const mjData* d, const Scan& scan, const GaitParams& params, double phase,
                                         double target) const;
  void rollout(const Scan& scan, const std::vector<GaitParams>& population, double target, double phase,
               std::vector<double>& costs, std::vector<bool>& failures);

  Sim& sim_;
  GaitConfig cfg_;
  NativeMethod method_;
  double fd_epsilon_, trust_radius_;
  GaitParams mean_{};
  double phase_ = 0;
  bool has_last_time_ = false;
  double last_time_ = 0;
  double last_plan_;
  double first_hold_;
  NormalSampler randn_;
  ControllerCounters counters_;
  ControllerDiagnostics diag_;
};

}  // namespace tpht
