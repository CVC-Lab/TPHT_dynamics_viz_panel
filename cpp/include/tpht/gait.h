// Gait references and the acceleration target. Port of src/controllers/gait.js
// (pure_cpht/controller.py::Controller.references and
// mujoco_reference/controller.py::NativeController.desired).
#pragma once

#include <array>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "tpht/linalg.h"

namespace tpht {

constexpr int kGaitParams = 19;  // offset(6), sin(6), cos(6), frequency_hz
using GaitParams = std::array<double, kGaitParams>;
using JointLimits = std::array<std::array<double, 2>, 6>;

// The fields of gait.json's "options" that the controllers read.
struct ControllerOptions {
  double speed_kd = 0, height_kp = 0, height_kd = 0, pitch_kp = 0, pitch_kd = 0;
  double joint_kp = 0, joint_kd = 0, target_clearance = 0;
  double feedback_ridge = 0;
  // MPC (mujoco_mpc); absent from feedback-only exports.
  double planning_dt = 0.01, replan_seconds = 0;
  int horizon_steps = 0, iterations = 0, candidates = 0, elites = 0;
  double speed_weight = 0, clearance_weight = 0, pitch_weight = 0, effort_weight = 0;
  double uncertainty_weight = 0, progress_weight = 0;

  static ControllerOptions from_json(const nlohmann::json& j);
};

// gait.json, exported by tools/export_web_assets.py gait: the saved gait and its search box.
struct GaitConfig {
  GaitParams gait{}, low{}, high{}, scale{};
  JointLimits limits{};
  ControllerOptions options;

  static GaitConfig from_json(const nlohmann::json& j);
};

struct References {
  std::array<double, 6> q{}, v{}, a{};
};

References references(const GaitParams& params, double phase, const JointLimits& limits);

// One scan row: [x, height, slope].
using ScanRow = std::array<double, 3>;
using Scan = std::vector<ScanRow>;

// What desiredAcceleration reads from MuJoCo after mj_forward + mj_subtreeVel.
struct TorsoInput {
  const double* qpos = nullptr;  // nq
  const double* qvel = nullptr;  // nv
  double torso_pos[3] = {0, 0, 0};
  double torso_linvel_x = 0;
};

std::array<double, 9> desired_acceleration(const TorsoInput& state, const Scan& scan, const GaitParams& params,
                                           double phase, double target, const ControllerOptions& o,
                                           const JointLimits& limits);

// Ground profile exported in the model metadata: z(x).
struct Profile {
  Vec x{-1e3, 1e3}, z{0, 0};
  static Profile from_json(const nlohmann::json& j);
};

// Scan offsets observed in validation/reference_samples (relative to q_rootx).
extern const std::array<double, 7> kScanOffsets;

// Privileged scan from the exported ground profile. The causal environment
// observer (environment_belief/observer.py) is not ported yet.
Scan privileged_scan(double root_x, const Profile& profile);

}  // namespace tpht
