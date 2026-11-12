// The numerics against the same fixtures and tolerances as tests/controllers.test.js
// (fixtures: tools/make_test_fixtures.py, from scipy and the Python formulas).
#include <cmath>
#include <fstream>
#include <sstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "tpht/bvls.h"
#include "tpht/gait.h"
#include "tpht/linalg.h"
#include "tpht/rng.h"

using nlohmann::json;

namespace {

json fixture(const char* name) {
  std::ifstream f(std::string(TPHT_SOURCE_DIR) + "/tests/fixtures/" + name);
  if (!f) throw std::runtime_error(std::string("missing fixture ") + name);
  std::stringstream s;
  s << f.rdbuf();
  return json::parse(s.str());
}

}  // namespace

TEST(BoundedLs, MatchesScipyBvlsOn400CheetahShapedProblems) {
  const json cases = fixture("bvls_cases.json");
  ASSERT_GE(cases.size(), 400u);
  double worst = 0, worst_res = 0;
  for (const auto& c : cases) {
    const auto J = c.at("J").get<std::vector<double>>();
    const auto offset = c.at("offset").get<std::vector<double>>();
    const auto desired = c.at("desired").get<std::vector<double>>();
    const auto low = c.at("low").get<std::vector<double>>();
    const auto high = c.at("high").get<std::vector<double>>();
    const auto x = c.at("x").get<std::vector<double>>();
    const auto r = tpht::bounded_ls(J.data(), offset.data(), desired.data(), 0.01, low.data(), high.data());
    for (int i = 0; i < 6; i++) worst = std::max(worst, std::abs(r.action[i] - x[i]));
    const double res = c.at("residual").get<double>();
    worst_res = std::max(worst_res, std::abs(r.diagnostics.weighted_residual - res) / std::max(1.0, res));
    EXPECT_LT(r.diagnostics.projected_gradient_inf, 1e-8) << "KKT residual";
  }
  EXPECT_LT(worst, 1e-7) << "max |x_cpp - x_scipy|";
  EXPECT_LT(worst_res, 1e-9) << "residual mismatch";
}

TEST(Gait, ReferencesAndAccelerationTargetMatchThePythonFormulas) {
  const json f = fixture("gait_cases.json");
  json opts = f.at("options");
  const tpht::ControllerOptions options = tpht::ControllerOptions::from_json(opts);
  tpht::JointLimits limits{};
  for (int i = 0; i < 6; i++) limits[i] = {f["limits"][i][0].get<double>(), f["limits"][i][1].get<double>()};
  double worst = 0;
  for (const auto& c : f.at("cases")) {
    tpht::GaitParams p{};
    for (int i = 0; i < tpht::kGaitParams; i++) p[i] = c["p"][i].get<double>();
    const double phase = c.at("phase").get<double>();
    const auto r = tpht::references(p, phase, limits);
    const auto qpos = c.at("qpos").get<std::vector<double>>();
    const auto qvel = c.at("qvel").get<std::vector<double>>();
    tpht::TorsoInput in;
    in.qpos = qpos.data();
    in.qvel = qvel.data();
    for (int i = 0; i < 3; i++) in.torso_pos[i] = c["root"][i].get<double>();
    in.torso_linvel_x = c.at("linvel_x").get<double>();
    tpht::Scan scan;
    for (const auto& row : c.at("scan")) scan.push_back({row[0].get<double>(), row[1].get<double>(), row[2].get<double>()});
    const auto d = tpht::desired_acceleration(in, scan, p, phase, 3, options, limits);
    for (int i = 0; i < 6; i++) {
      worst = std::max({worst, std::abs(r.q[i] - c["qr"][i].get<double>()), std::abs(r.v[i] - c["vr"][i].get<double>()),
                        std::abs(r.a[i] - c["ar"][i].get<double>())});
    }
    for (int i = 0; i < 9; i++) {
      const double want = c["desired"][i].get<double>();
      worst = std::max(worst, std::abs(d[i] - want) / std::max(1.0, std::abs(want)));
    }
  }
  EXPECT_LT(worst, 1e-10);
}

TEST(Linalg, InterpClampsLikeNumpy) {
  const tpht::Vec xs{0, 1}, ys{2, 4};
  EXPECT_EQ(tpht::interp(-5, xs, ys), 2);
  EXPECT_EQ(tpht::interp(5, xs, ys), 4);
  EXPECT_EQ(tpht::interp(0.25, xs, ys), 2.5);
}

TEST(Linalg, CholeskyRejectsIndefiniteAndNaN) {
  const double indefinite[4] = {1, 2, 2, 1};
  const double nan[4] = {NAN, 0, 0, 1};
  double l[4];
  EXPECT_FALSE(tpht::cholesky(indefinite, 2, l));
  EXPECT_FALSE(tpht::cholesky(nan, 2, l));
}

// Reference values from an independent Python port of mulberry32 (integer
// arithmetic, so exact); tests/controllers.test.js checks the JavaScript
// generator against the same numbers.
TEST(Rng, Mulberry32MatchesTheJavaScriptGenerator) {
  tpht::Mulberry32 u(100000);
  const double want[] = {0.45954178320243955, 0.7448947750963271, 0.2210056281182915,
                         0.47198858694173396, 0.059644044609740376, 0.3554476269055158};
  for (const double w : want) EXPECT_EQ(u.uniform(), w);
  tpht::NormalSampler n(100000);
  const double normal[] = {-0.03999378991677011, -1.7107233084267035, -1.460680052765154, -0.42671675085463895};
  for (const double w : normal) EXPECT_NEAR(n(), w, 1e-15);
}
