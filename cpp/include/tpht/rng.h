// The seeded normal sampler of src/controllers/native.js (mulberry32 + Box-Muller),
// in exact integer arithmetic, so mujoco_mpc draws the same candidates as the
// JavaScript backend. (Both differ from the Python arm, which uses torch.Generator.)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tpht {

class Mulberry32 {
 public:
  explicit Mulberry32(uint32_t seed) : a_(seed) {}
  // Uniform in [0, 1), as ((t ^ (t >>> 14)) >>> 0) / 2^32 in JavaScript.
  double uniform() {
    a_ += 0x6d2b79f5u;
    uint32_t t = a_;
    t = (t ^ (t >> 15)) * (t | 1u);
    t ^= t + (t ^ (t >> 7)) * (t | 61u);
    return static_cast<double>(t ^ (t >> 14)) / 4294967296.0;
  }

 private:
  uint32_t a_;
};

class NormalSampler {
 public:
  explicit NormalSampler(uint32_t seed) : u_(seed) {}
  double operator()() {
    const double u = std::max(u_.uniform(), 1e-300), v = u_.uniform();
    return std::sqrt(-2 * std::log(u)) * std::cos(2 * 3.14159265358979323846 * v);
  }

 private:
  Mulberry32 u_;
};

}  // namespace tpht
