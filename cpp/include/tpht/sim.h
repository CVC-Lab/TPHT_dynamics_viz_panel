// MuJoCo model + data for one run. Port of src/backend/mujoco_env.js (class Sim)
// onto MuJoCo's C API, so the page's WebAssembly build and the native tools share
// one implementation.
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <nlohmann/json_fwd.hpp>

#include "tpht/linalg.h"

namespace tpht {

// An error with a stable code, shown by the page like mujoco_env.js's BackendError.
class BackendError : public std::runtime_error {
 public:
  BackendError(std::string code, const std::string& message) : std::runtime_error(message), code_(std::move(code)) {}
  const std::string& code() const { return code_; }

 private:
  std::string code_;
};

// A model file: kind "xml" or "mjb", its file name, and its bytes.
struct ModelAsset {
  std::string kind;
  std::string name;
  std::string data;
};

struct SelfTest {
  std::string name;
  bool ok = false;
  std::string detail;
};

struct SelfTestExpect {
  std::optional<std::array<int, 3>> dims;  // nq, nv, nu
  std::optional<double> timestep;
  bool euler = false;
};

struct TorsoState {
  const double* qpos = nullptr;
  const double* qvel = nullptr;
  double torso_pos[3] = {0, 0, 0};
  double torso_linvel_x = 0;
  double torso_com_x = 0;
};

struct ModelDeleter { void operator()(mjModel* m) const { mj_deleteModel(m); } };
struct DataDeleter { void operator()(mjData* d) const { mj_deleteData(d); } };
using ModelPtr = std::unique_ptr<mjModel, ModelDeleter>;
using DataPtr = std::unique_ptr<mjData, DataDeleter>;

// Compiles an XML model (through a one-file VFS) or loads an .mjb. *via names
// the loader used. Throws BackendError.
ModelPtr load_model(const ModelAsset& asset, std::string* via = nullptr);

class Sim {
 public:
  // meta: the exported model metadata (torso_body_id, legacy_step, ...); missing
  // fields fall back to name lookups and defaults, as in mujoco_env.js.
  Sim(ModelPtr model, const nlohmann::json& meta);

  mjModel* model() const { return model_.get(); }
  mjData* data() const { return data_.get(); }
  mjData* scratch(int i) const { return scratch_.at(i).get(); }
  int nv() const { return nv_; }
  int nu() const { return nu_; }
  int torso() const { return torso_; }
  bool legacy_step() const { return legacy_step_; }
  // Generalized actuation map (nv x nu): 3 unactuated root coordinates, then diag(gear).
  const Vec& B() const { return B_; }

  // mj_forward + mj_subtreeVel, after setting ctrl when action is given (nu values).
  void forward(mjData* d, const double* action = nullptr) const;
  mjData* copy_into(mjData* dst, const mjData* src) const;
  void reset_data();  // mj_resetData + forward, the CT-RL tab's reset
  void set_state(const double* q, const double* v, const double* ctrl);
  // One control hold: dm_control's legacy split step (mj_step2 then mj_step1),
  // or mj_step when the metadata says legacy_step: false.
  void step_hold(const double* action, int substeps);

  // Planar push on the torso COM: world-frame force (fx, fz) in N and pitch torque ty in N m.
  void push(double fx, double fz, double ty, double duration);
  void clear_push();
  bool push_active() const;
  // Scales the sliding friction of every geom.
  void set_friction_scale(double s);

  Vec full_mass(const mjData* d) const;  // nv x nv, row-major
  TorsoState torso_state(const mjData* d) const;

  // Robot geoms projected onto the x-z plane: [x1, z1, x2, z2, radius] per geom.
  std::vector<float> segments(const mjData* d) const;
  std::vector<float> ghost_segments(const double* q);

  std::vector<SelfTest> self_test(const SelfTestExpect& expect);

 private:
  int name_to_id(int type, const char* name) const;

  ModelPtr model_;
  DataPtr data_, ghost_;
  std::array<DataPtr, 5> scratch_;
  int nv_ = 0, nu_ = 0, torso_ = 0;
  Vec gear_, B_, friction_base_;
  double push_until_;
  bool legacy_step_ = true;
};

}  // namespace tpht
