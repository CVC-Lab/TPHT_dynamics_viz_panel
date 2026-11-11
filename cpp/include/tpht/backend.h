// One loaded model, its state and the active controller: what the page's worker
// (src/backend/sim.worker.js) and the native tool (tpht-sim) drive. The worker
// keeps the clock, messages, traces and statistics; this holds the simulation and
// the decisions, so the WebAssembly build and the native build run the same code.
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tpht/gait.h"
#include "tpht/native_controller.h"
#include "tpht/sim.h"

namespace tpht {

// The built-in open-loop gait of the CT-RL tab (src/controllers/demo_gait.js).
// Not a baseline: fixed sinusoids on the six actuators, the front leg a quarter
// cycle behind the back leg. It reads only the simulation time.
class SineGait {
 public:
  static constexpr const char* kId = "demo-sine";
  explicit SineGait(int nu, double frequency = 2.5, double amplitude = 0.7, double joint_lag = 0.6,
                    double leg_offset = 3.14159265358979323846 / 2);
  std::vector<float> act(double time) const;
  int nu() const { return static_cast<int>(phase_.size()); }

 private:
  double frequency_, amplitude_;
  std::vector<double> phase_;
};

struct LoadResult {
  std::string via;
  std::vector<SelfTest> self_tests;  // the model loader first, as the page lists them
};

struct GoldenCheck {
  std::string name;
  std::optional<double> value, tol;
  std::string note;
};

class Backend {
 public:
  static std::string mujoco_version();

  // Loads a model and runs the binding self-tests. meta: the model metadata JSON
  // (may be empty). Throws BackendError.
  LoadResult load(const ModelAsset& asset, const std::string& meta_json, const SelfTestExpect& expect);
  bool loaded() const { return sim_ != nullptr; }
  Sim& sim();

  // gait.json for the reference controllers; an empty string clears it.
  void set_gait(const std::string& gait_json);
  bool has_gait() const { return gait_.has_value(); }
  const Profile& profile() const { return profile_; }

  void reset_data();                                                // CT-RL: the model's qpos0
  void set_state(const double* q, const double* v, const double* ctrl);  // reference: a trace row

  void use_native(const std::string& method, uint32_t seed = 100000);  // needs set_gait first
  void use_sine_gait();
  void clear_controller();
  bool has_controller() const { return native_ != nullptr || sine_ != nullptr; }

  // The active controller's action for the current state (does not step).
  // Native controllers read the privileged scan of the exported ground profile.
  std::vector<float> decide(double target, double hold);
  // Scalar diagnostics of the last decision, with the keys the page shows.
  nlohmann::json diagnostics() const;

  void step_hold(const float* action, int n, int substeps);
  void push(double fx, double fz, double ty, double duration) { sim().push(fx, fz, ty, duration); }
  void set_friction_scale(double s) { sim().set_friction_scale(s); }

  // Replays validation/reference_samples/<clock>/<case>_transition.json: the
  // recorded action through the model, then (with a gait) the acceleration
  // target and the fitted action. Leaves the data reset.
  std::vector<GoldenCheck> golden(const std::string& sample_json);

 private:
  std::unique_ptr<Sim> sim_;
  nlohmann::json meta_;
  Profile profile_;
  std::optional<GaitConfig> gait_;
  std::unique_ptr<NativeController> native_;
  std::unique_ptr<SineGait> sine_;
  double sine_decision_ms_ = 0;
};

nlohmann::json to_json(const std::vector<SelfTest>& tests);
nlohmann::json to_json(const std::vector<GoldenCheck>& checks);

}  // namespace tpht
