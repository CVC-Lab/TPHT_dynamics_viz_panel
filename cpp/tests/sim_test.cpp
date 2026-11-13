// The backend on dm_control's cheetah (public/assets/ctrl), the model the page's
// CT-RL tab loads. The reference cases need the benchmark's models, which are
// git-ignored; the controllers are exercised here on the cheetah instead, with
// the options of tests/fixtures/gait_cases.json.
#include <cmath>
#include <fstream>
#include <sstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "tpht/backend.h"

using nlohmann::json;

namespace {

std::string read(const std::string& rel) {
  std::ifstream f(std::string(TPHT_SOURCE_DIR) + "/" + rel, std::ios::binary);
  if (!f) throw std::runtime_error("missing " + rel);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

tpht::Backend load_cheetah() {
  tpht::Backend b;
  tpht::ModelAsset asset{"xml", "cheetah.xml", read("public/assets/ctrl/cheetah.xml")};
  tpht::SelfTestExpect expect;
  expect.timestep = 0.01;
  b.load(asset, read("public/assets/ctrl/cheetah_meta.json"), expect);
  return b;
}

// A gait.json for the cheetah (tests only): the feedback gains of gait_cases.json,
// a mild trot, and small MPC settings so the tests stay fast. tests/backend.test.js
// runs the same file through both browser backends.
std::string cheetah_gait_json() { return read("tests/fixtures/cheetah_gait.json"); }

// Values that depend on the MuJoCo release, keyed by version (also read by
// tests/backend.test.js).
double mujoco_reference(const std::string& key) {
  const json ref = json::parse(read("tests/fixtures/mujoco_reference.json"));
  const std::string v = tpht::Backend::mujoco_version();
  if (!ref.at(key).contains(v)) throw std::runtime_error("no " + key + " for MuJoCo " + v);
  return ref.at(key).at(v).get<double>();
}

}  // namespace

TEST(Sim, CheetahSelfTestsPass) {
  tpht::Backend b;
  tpht::ModelAsset asset{"xml", "cheetah.xml", read("public/assets/ctrl/cheetah.xml")};
  tpht::SelfTestExpect expect;
  expect.timestep = 0.01;
  const auto r = b.load(asset, read("public/assets/ctrl/cheetah_meta.json"), expect);
  ASSERT_GE(r.self_tests.size(), 5u);
  for (const auto& t : r.self_tests) EXPECT_TRUE(t.ok) << t.name << ": " << t.detail;
  EXPECT_EQ(r.via, "mj_loadXML (VFS)");
  EXPECT_EQ(tpht::Backend::mujoco_version(), TPHT_MUJOCO_VERSION);
}

TEST(Sim, BadModelsFailWithCodes) {
  tpht::Backend b;
  try {
    b.load({"xml", "broken.xml", "<mujoco><worldbody><body"}, "", {});
    FAIL() << "expected a compile error";
  } catch (const tpht::BackendError& e) {
    EXPECT_EQ(e.code(), "model-compile");
  }
  try {
    b.load({"mjb", "old.mjb", std::string(64, '\0')}, "", {});
    FAIL() << "expected an mjb error";
  } catch (const tpht::BackendError& e) {
    EXPECT_EQ(e.code(), "mjb-load");
  }
}

// The same rollout with the MuJoCo Python package of the same version, stepping
// as the page does (float32 controls, mj_step2 then mj_step1 from a forwarded
// reset):
//   d.ctrl[:] = float32(sine_gait(nu, d.time)); mj_step2; mj_step1   (500 times)
// The distance qpos[0] moves is in tests/fixtures/mujoco_reference.json
// ("demo_dx_5s"), per version. tpht_lab/check.py reports the torso's xpos instead.
TEST(Sim, DemoGaitMatchesTheNativePythonRollout) {
  tpht::Backend b = load_cheetah();
  b.reset_data();
  b.use_sine_gait();
  tpht::Sim& s = b.sim();
  const double x0 = s.data()->qpos[0];
  double max_pitch = 0;
  for (int k = 0; k < 500; k++) {
    const auto a = b.decide(3, 0.01);
    b.step_hold(a.data(), 6, 1);
    max_pitch = std::max(max_pitch, std::abs(s.data()->qpos[2]));
  }
  EXPECT_NEAR(s.data()->time, 5.0, 1e-9);
  EXPECT_NEAR(s.data()->qpos[0] - x0, mujoco_reference("demo_dx_5s"), 1e-9);
  EXPECT_LT(max_pitch, 0.3);
}

TEST(Sim, SegmentsAndGhost) {
  tpht::Backend b = load_cheetah();
  b.reset_data();
  tpht::Sim& s = b.sim();
  const auto segs = s.segments(s.data());
  ASSERT_EQ(segs.size() % 5, 0u);
  EXPECT_GT(segs.size(), 0u);
  const auto ghost = s.ghost_segments(s.data()->qpos);
  ASSERT_EQ(ghost.size(), segs.size());
  for (size_t i = 0; i < segs.size(); i++) EXPECT_NEAR(ghost[i], segs[i], 1e-6);
}

TEST(Sim, PushExpiresAfterItsDuration) {
  tpht::Backend b = load_cheetah();
  b.reset_data();
  b.use_sine_gait();
  tpht::Sim& s = b.sim();
  s.push(200, 0, 0, 0.05);
  EXPECT_TRUE(s.push_active());
  for (int k = 0; k < 10; k++) {
    const auto a = b.decide(3, 0.01);
    b.step_hold(a.data(), 6, 1);
  }
  EXPECT_FALSE(s.push_active());
  for (int i = 0; i < 6; i++) EXPECT_EQ(s.data()->xfrc_applied[s.torso() * 6 + i], 0);
}

TEST(NativeController, FeedbackSolvesAreBoundedAndConsistent) {
  for (const char* method : {"mujoco_ls", "mujoco_fd_ls"}) {
    tpht::Backend b = load_cheetah();
    b.set_gait(cheetah_gait_json());
    b.reset_data();
    b.use_native(method);
    for (int k = 0; k < 50; k++) {
      const auto a = b.decide(1.0, 0.01);
      for (const float x : a) {
        EXPECT_LE(std::abs(x), 1.0f) << method;
      }
      const json d = b.diagnostics();
      EXPECT_LT(d["ls_projected_gradient_inf"].get<double>(), 1e-8) << method;
      b.step_hold(a.data(), 6, 1);
    }
    for (int i = 0; i < 9; i++) EXPECT_TRUE(std::isfinite(b.sim().data()->qpos[i])) << method;
  }
}

// The analytic acceleration map (mujoco_ls) and the finite-difference one
// (mujoco_fd_ls) describe the same linear response away from contact changes.
TEST(NativeController, AnalyticAndFiniteDifferenceMapsAgreeInTheAir) {
  tpht::Backend b = load_cheetah();
  b.set_gait(cheetah_gait_json());
  b.reset_data();
  tpht::Sim& s = b.sim();
  s.data()->qpos[1] = 1.0;  // lift the torso clear of the floor
  s.forward(s.data());
  const tpht::GaitConfig cfg = tpht::GaitConfig::from_json(json::parse(cheetah_gait_json()));
  tpht::NativeController ls(s, cfg, tpht::NativeMethod::ls), fd(s, cfg, tpht::NativeMethod::fd_ls);
  const tpht::Scan scan = tpht::privileged_scan(s.data()->qpos[0], b.profile());
  const auto a = ls.fit(s.copy_into(s.scratch(0), s.data()), scan, ls.mean(), 0.3, 1.0, false, true);
  const auto f = fd.fit(s.copy_into(s.scratch(0), s.data()), scan, fd.mean(), 0.3, 1.0, true, true);
  for (int i = 0; i < 9; i++) EXPECT_NEAR(a.acceleration_target[i], f.acceleration_target[i], 1e-12);
  EXPECT_LT(a.contact_linearization_error, 1e-6);
  EXPECT_LT(f.contact_linearization_error, 1e-3);
}

TEST(NativeController, MpcPlansOnSchedule) {
  tpht::Backend b = load_cheetah();
  b.set_gait(cheetah_gait_json());
  b.reset_data();
  b.use_native("mujoco_mpc", 7);
  int plans = 0;
  for (int k = 0; k < 41; k++) {
    const auto a = b.decide(1.0, 0.01);
    const json d = b.diagnostics();
    plans += d["planned"].get<bool>();
    EXPECT_TRUE(d.contains("planner_cost"));
    b.step_hold(a.data(), 6, 1);
  }
  EXPECT_EQ(plans, 3);  // t = 0, 0.2, 0.4 with replan_seconds 0.2
}

TEST(NativeController, RejectsOtherModelShapesAndMissingGait) {
  tpht::Backend b = load_cheetah();
  try {
    b.use_native("mujoco_ls");
    FAIL() << "expected gait-missing";
  } catch (const tpht::BackendError& e) {
    EXPECT_EQ(e.code(), "gait-missing");
  }
  b.set_gait(cheetah_gait_json());
  EXPECT_THROW(b.use_native("mujoco_nope"), tpht::BackendError);
}
