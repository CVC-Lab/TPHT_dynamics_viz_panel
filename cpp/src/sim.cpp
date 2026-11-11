#include "tpht/sim.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include <nlohmann/json.hpp>

namespace tpht {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

std::string fmt(const char* format, double x) {
  char buf[64];
  std::snprintf(buf, sizeof buf, format, x);
  return buf;
}

}  // namespace

ModelPtr load_model(const ModelAsset& asset, std::string* via) {
  if (asset.kind == "xml") {
    // A one-file VFS, so the bytes never have to touch a real file system.
    mjVFS vfs;
    mj_defaultVFS(&vfs);
    const int added = mj_addBufferVFS(&vfs, asset.name.c_str(), asset.data.data(), static_cast<int>(asset.data.size()));
    if (added != 0) {
      mj_deleteVFS(&vfs);
      throw BackendError("model-load", "Could not add " + asset.name + " to MuJoCo's virtual file system");
    }
    char error[1024] = "";
    mjModel* m = mj_loadXML(asset.name.c_str(), &vfs, error, sizeof error);
    mj_deleteVFS(&vfs);
    if (!m) throw BackendError("model-compile", "MuJoCo could not compile " + asset.name + ": " + error);
    if (via) *via = "mj_loadXML (VFS)";
    return ModelPtr(m);
  }
  if (asset.kind == "mjb") {
    mjModel* m = mj_loadModelBuffer(asset.data.data(), static_cast<mjtSize>(asset.data.size()));
    if (!m) {
      throw BackendError("mjb-load", "MuJoCo " + std::string(mj_versionString()) + " could not load " + asset.name +
                                         ". An .mjb only loads in the MuJoCo version that wrote it; export XML instead "
                                         "(see the README).");
    }
    if (via) *via = "mj_loadModelBuffer";
    return ModelPtr(m);
  }
  throw BackendError("model-kind", "Unknown model kind \"" + asset.kind + "\" (expected xml or mjb)");
}

Sim::Sim(ModelPtr model, const nlohmann::json& meta)
    : model_(std::move(model)), push_until_(kNegInf) {
  mjModel* m = model_.get();
  data_.reset(mj_makeData(m));
  ghost_.reset(mj_makeData(m));
  for (auto& d : scratch_) d.reset(mj_makeData(m));
  nv_ = m->nv;
  nu_ = m->nu;
  const bool has_meta = meta.is_object();
  if (has_meta && meta.contains("torso_body_id") && !meta["torso_body_id"].is_null()) {
    torso_ = meta["torso_body_id"].get<int>();
  } else {
    torso_ = name_to_id(mjOBJ_BODY, "torso");
  }
  gear_.resize(nu_);
  for (int i = 0; i < nu_; i++) gear_[i] = m->actuator_gear[i * 6];
  B_.assign(static_cast<size_t>(nv_) * nu_, 0.0);
  for (int i = 0; i < nu_; i++) B_[(nv_ - nu_ + i) * nu_ + i] = gear_[i];
  friction_base_.resize(m->ngeom);
  for (int g = 0; g < m->ngeom; g++) friction_base_[g] = m->geom_friction[g * 3];
  legacy_step_ = has_meta && meta.contains("legacy_step") && !meta["legacy_step"].is_null()
                     ? meta["legacy_step"].get<bool>()
                     : true;
}

int Sim::name_to_id(int type, const char* name) const {
  const int id = mj_name2id(model_.get(), type, name);
  if (id < 0) {
    throw BackendError("name-lookup", std::string("Cannot resolve \"") + name + "\"; add it to the exported metadata.");
  }
  return id;
}

void Sim::forward(mjData* d, const double* action) const {
  if (action) std::copy(action, action + nu_, d->ctrl);
  mj_forward(model_.get(), d);
  mj_subtreeVel(model_.get(), d);
}

mjData* Sim::copy_into(mjData* dst, const mjData* src) const {
  mj_copyData(dst, model_.get(), src);
  return dst;
}

void Sim::reset_data() {
  mj_resetData(model_.get(), data_.get());
  push_until_ = kNegInf;
  forward(data_.get());
}

void Sim::set_state(const double* q, const double* v, const double* ctrl) {
  mjModel* m = model_.get();
  mjData* d = data_.get();
  mj_resetData(m, d);
  std::copy(q, q + m->nq, d->qpos);
  std::copy(v, v + m->nv, d->qvel);
  if (ctrl) std::copy(ctrl, ctrl + m->nu, d->ctrl);
  push_until_ = kNegInf;
  // After mj_forward the split-step invariant (step1 done) holds.
  forward(d);
}

void Sim::step_hold(const double* action, int substeps) {
  mjModel* m = model_.get();
  mjData* d = data_.get();
  std::copy(action, action + nu_, d->ctrl);
  for (int k = 0; k < substeps; k++) {
    if (d->time >= push_until_) clear_push();
    if (legacy_step_) { mj_step2(m, d); mj_step1(m, d); }
    else mj_step(m, d);
  }
  mj_subtreeVel(m, d);
}

void Sim::push(double fx, double fz, double ty, double duration) {
  double* x = data_->xfrc_applied + torso_ * 6;
  x[0] = fx; x[1] = 0; x[2] = fz; x[3] = 0; x[4] = ty; x[5] = 0;
  push_until_ = data_->time + duration;
}

void Sim::clear_push() {
  double* x = data_->xfrc_applied + torso_ * 6;
  std::fill(x, x + 6, 0.0);
  push_until_ = kNegInf;
}

bool Sim::push_active() const { return data_->time < push_until_; }

void Sim::set_friction_scale(double s) {
  for (size_t g = 0; g < friction_base_.size(); g++) model_->geom_friction[g * 3] = friction_base_[g] * s;
}

Vec Sim::full_mass(const mjData* d) const {
  Vec M(static_cast<size_t>(nv_) * nv_);
  mj_fullM(model_.get(), d, M.data());
  return M;
}

TorsoState Sim::torso_state(const mjData* d) const {
  TorsoState s;
  const int t = torso_;
  s.qpos = d->qpos;
  s.qvel = d->qvel;
  s.torso_pos[0] = d->xpos[t * 3];
  s.torso_pos[1] = d->xpos[t * 3 + 1];
  s.torso_pos[2] = d->xpos[t * 3 + 2];
  s.torso_linvel_x = d->subtree_linvel[t * 3];
  s.torso_com_x = d->subtree_com[t * 3];
  return s;
}

std::vector<float> Sim::segments(const mjData* d) const {
  const mjModel* m = model_.get();
  std::vector<float> out;
  out.reserve(static_cast<size_t>(m->ngeom) * 5);
  auto push5 = [&](double a, double b, double c, double e, double r) {
    out.push_back(static_cast<float>(a)); out.push_back(static_cast<float>(b));
    out.push_back(static_cast<float>(c)); out.push_back(static_cast<float>(e));
    out.push_back(static_cast<float>(r));
  };
  for (int g = 0; g < m->ngeom; g++) {
    if (m->geom_bodyid[g] == 0) continue;
    const int type = m->geom_type[g];
    const double px = d->geom_xpos[g * 3], pz = d->geom_xpos[g * 3 + 2];
    const double* R = d->geom_xmat + g * 9;
    const double s0 = m->geom_size[g * 3], s1 = m->geom_size[g * 3 + 1];
    if (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) {  // local z axis
      const double ax = R[2] * s1, az = R[8] * s1;
      push5(px - ax, pz - az, px + ax, pz + az, s0);
    } else if (type == mjGEOM_BOX) {  // drawn along local x
      const double ax = R[0] * s0, az = R[6] * s0;
      push5(px - ax, pz - az, px + ax, pz + az, m->geom_size[g * 3 + 2]);
    } else {  // sphere, ellipsoid, mesh: a dot
      push5(px, pz, px, pz, s0);
    }
  }
  return out;
}

std::vector<float> Sim::ghost_segments(const double* q) {
  std::copy(q, q + model_->nq, ghost_->qpos);
  mj_kinematics(model_.get(), ghost_.get());
  return segments(ghost_.get());
}

std::vector<SelfTest> Sim::self_test(const SelfTestExpect& expect) {
  std::vector<SelfTest> results;
  auto add = [&](std::string name, bool ok, std::string detail) {
    results.push_back({std::move(name), ok, std::move(detail)});
  };
  const mjModel* m = model_.get();
  try {
    if (expect.dims) {
      const auto& e = *expect.dims;
      add("Model dimensions", m->nq == e[0] && m->nv == e[1] && m->nu == e[2],
          "nq=" + std::to_string(m->nq) + " nv=" + std::to_string(m->nv) + " nu=" + std::to_string(m->nu) +
              ", expected " + std::to_string(e[0]) + "/" + std::to_string(e[1]) + "/" + std::to_string(e[2]));
    }
    if (expect.timestep) {
      add("Physics timestep", std::abs(m->opt.timestep - *expect.timestep) < 1e-12,
          fmt("%g s", m->opt.timestep) + fmt(", expected %g s", *expect.timestep));
    }
    if (expect.euler) {
      add("Euler integrator", m->opt.integrator == mjINT_EULER, "integrator=" + std::to_string(m->opt.integrator));
    }
    bool bounds_ok = true;
    std::string bounds = "ctrlrange=";
    for (int i = 0; i < 2 * m->nu; i++) {
      const double x = m->actuator_ctrlrange[i];
      bounds_ok = bounds_ok && std::abs(x - (i % 2 ? 1 : -1)) < 1e-12;
      bounds += (i ? " " : "") + fmt("%.2f", x);
    }
    add("Normalized actuator bounds", bounds_ok, bounds);

    mjData* probe = copy_into(scratch_[0].get(), data_.get());
    Vec ctrl(nu_);
    for (int i = 0; i < nu_; i++) ctrl[i] = -0.4 + (0.8 * i) / std::max(1, nu_ - 1);
    forward(probe, ctrl.data());
    double gear_err = 0;
    for (int r = 0; r < nv_; r++) {
      double s = 0;
      for (int c = 0; c < nu_; c++) s += B_[r * nu_ + c] * ctrl[c];
      gear_err = std::max(gear_err, std::abs(probe->qfrc_actuator[r] - s));
    }
    add("Linear gear map (qfrc_actuator = B ctrl)", gear_err < 1e-9, fmt("max error %.2e", gear_err));
    add("State copy", true, "mj_copyData");

    const Vec M = full_mass(probe);
    double asym = 0, balance = 0, scale = 1;
    for (int i = 0; i < nv_; i++) {
      double s = 0;
      for (int j = 0; j < nv_; j++) {
        asym = std::max(asym, std::abs(M[i * nv_ + j] - M[j * nv_ + i]));
        s += M[i * nv_ + j] * probe->qacc[j];
      }
      const double f = probe->qfrc_actuator[i] + probe->qfrc_passive[i] - probe->qfrc_bias[i] +
                       probe->qfrc_constraint[i] + probe->qfrc_applied[i];
      scale = std::max(scale, std::abs(f));
      balance = std::max(balance, std::abs(s - f));
    }
    add("Mass matrix via mj_fullM", asym < 1e-9 && balance / scale < 1e-8,
        fmt("asymmetry %.2e", asym) + fmt(", force balance %.2e (relative)", balance / scale));
  } catch (const std::exception& e) {
    add("Binding self-test", false, e.what());
  }
  return results;
}

}  // namespace tpht
