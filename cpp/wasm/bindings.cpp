// Embind bindings: the C++ backend as the JavaScript class TphtBackend, for the
// page's Web Worker (src/backend/cpp_backend.js wraps it). Errors reach
// JavaScript as Error objects with the BackendError code in .code, like
// mujoco_env.js's BackendError. Arrays cross as typed-array copies.
#include <cstdint>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "tpht/backend.h"

using emscripten::val;

namespace {

[[noreturn]] void throw_js(const std::string& code, const std::string& message) {
  val err = val::global("Error").new_(val(message));
  err.set("code", val(code));
  err.throw_();
  __builtin_unreachable();
}

template <typename F>
auto guard(F&& f) -> decltype(f()) {
  try {
    return f();
  } catch (const tpht::BackendError& e) {
    throw_js(e.code(), e.what());
  } catch (const std::exception& e) {
    throw_js("backend", e.what());
  }
}

std::string bytes_from(const val& input) {
  if (input.isString()) return input.as<std::string>();
  const size_t n = input["length"].as<size_t>();
  std::string out(n, '\0');
  val(emscripten::typed_memory_view(n, reinterpret_cast<uint8_t*>(out.data()))).call<void>("set", input);
  return out;
}

std::vector<double> doubles_from(const val& input) {
  return emscripten::convertJSArrayToNumberVector<double>(input);
}

template <typename T>
val copy_out(const char* ctor, const T* data, size_t n) {
  return val::global(ctor).new_(val(emscripten::typed_memory_view(n, data)));
}

class TphtBackend {
 public:
  static std::string mujocoVersion() { return tpht::Backend::mujoco_version(); }

  // expectJson: {"dims": [nq, nv, nu], "timestep": s, "euler": bool}, any part optional.
  std::string load(const std::string& kind, const std::string& name, const val& bytes, const std::string& meta_json,
                   const std::string& expect_json) {
    return guard([&] {
      tpht::SelfTestExpect expect;
      if (!expect_json.empty()) {
        const auto e = nlohmann::json::parse(expect_json);
        if (e.contains("dims") && e["dims"].is_array()) {
          expect.dims = std::array<int, 3>{e["dims"][0].get<int>(), e["dims"][1].get<int>(), e["dims"][2].get<int>()};
        }
        if (e.contains("timestep") && e["timestep"].is_number()) expect.timestep = e["timestep"].get<double>();
        expect.euler = e.value("euler", false);
      }
      const auto r = b_.load({kind, name, bytes_from(bytes)}, meta_json, expect);
      return nlohmann::json({{"via", r.via}, {"selfTests", tpht::to_json(r.self_tests)}}).dump();
    });
  }

  void setGait(const std::string& gait_json) { guard([&] { b_.set_gait(gait_json); }); }
  bool hasGait() const { return b_.has_gait(); }

  void resetData() { guard([&] { b_.reset_data(); }); }
  void setState(const val& q, const val& v, const val& ctrl) {
    guard([&] {
      const auto qq = doubles_from(q), vv = doubles_from(v);
      tpht::Sim& s = b_.sim();
      if (static_cast<int>(qq.size()) != s.model()->nq || static_cast<int>(vv.size()) != s.nv()) {
        throw tpht::BackendError("state-size", "setState: expected nq=" + std::to_string(s.model()->nq) + " and nv=" + std::to_string(s.nv()) + " values");
      }
      std::vector<double> uu;
      if (!ctrl.isNull() && !ctrl.isUndefined()) uu = doubles_from(ctrl);
      b_.set_state(qq.data(), vv.data(), uu.size() == static_cast<size_t>(s.nu()) ? uu.data() : nullptr);
    });
  }

  void useNative(const std::string& method, uint32_t seed) { guard([&] { b_.use_native(method, seed); }); }
  void useSineGait() { guard([&] { b_.use_sine_gait(); }); }
  void clearController() { b_.clear_controller(); }
  bool hasController() const { return b_.has_controller(); }

  val decide(double target, double hold) {
    return guard([&] {
      const auto a = b_.decide(target, hold);
      return copy_out("Float32Array", a.data(), a.size());
    });
  }
  std::string diagnosticsJson() const { return b_.diagnostics().dump(); }

  void stepHold(const val& action, int substeps) {
    guard([&] {
      const auto a = emscripten::convertJSArrayToNumberVector<float>(action);
      b_.step_hold(a.data(), static_cast<int>(a.size()), substeps);
    });
  }

  void push(double fx, double fz, double ty, double duration) { guard([&] { b_.push(fx, fz, ty, duration); }); }
  bool pushActive() { return guard([&] { return b_.sim().push_active(); }); }
  val pushForce() {
    return guard([&] {
      tpht::Sim& s = b_.sim();
      return copy_out("Float64Array", s.data()->xfrc_applied + s.torso() * 6, 6);
    });
  }
  void setFrictionScale(double s) { guard([&] { b_.set_friction_scale(s); }); }

  double time() { return guard([&] { return b_.sim().data()->time; }); }
  double timestep() { return guard([&] { return b_.sim().model()->opt.timestep; }); }
  int nq() { return guard([&] { return b_.sim().model()->nq; }); }
  int nv() { return guard([&] { return b_.sim().nv(); }); }
  int nu() { return guard([&] { return b_.sim().nu(); }); }
  val qpos() { return guard([&] { auto& s = b_.sim(); return copy_out("Float64Array", s.data()->qpos, s.model()->nq); }); }
  val qvel() { return guard([&] { auto& s = b_.sim(); return copy_out("Float64Array", s.data()->qvel, s.nv()); }); }
  // [torso x, torso z, torso COM-subtree linear velocity x, subtree COM x]
  val torso() {
    return guard([&] {
      auto& s = b_.sim();
      const auto st = s.torso_state(s.data());
      const double out[4] = {st.torso_pos[0], st.torso_pos[2], st.torso_linvel_x, st.torso_com_x};
      return copy_out("Float64Array", out, 4);
    });
  }
  val segments() {
    return guard([&] {
      auto& s = b_.sim();
      const auto v = s.segments(s.data());
      return copy_out("Float32Array", v.data(), v.size());
    });
  }
  val ghostSegments(const val& q) {
    return guard([&] {
      const auto qq = doubles_from(q);
      auto& s = b_.sim();
      if (static_cast<int>(qq.size()) < s.model()->nq) throw tpht::BackendError("state-size", "ghostSegments: need nq values");
      const auto v = s.ghost_segments(qq.data());
      return copy_out("Float32Array", v.data(), v.size());
    });
  }
  std::string goldenJson(const std::string& sample_json) {
    return guard([&] { return tpht::to_json(b_.golden(sample_json)).dump(); });
  }

 private:
  tpht::Backend b_;
};

}  // namespace

EMSCRIPTEN_BINDINGS(tpht_backend) {
  emscripten::class_<TphtBackend>("TphtBackend")
      .constructor<>()
      .class_function("mujocoVersion", &TphtBackend::mujocoVersion)
      .function("load", &TphtBackend::load)
      .function("setGait", &TphtBackend::setGait)
      .function("hasGait", &TphtBackend::hasGait)
      .function("resetData", &TphtBackend::resetData)
      .function("setState", &TphtBackend::setState)
      .function("useNative", &TphtBackend::useNative)
      .function("useSineGait", &TphtBackend::useSineGait)
      .function("clearController", &TphtBackend::clearController)
      .function("hasController", &TphtBackend::hasController)
      .function("decide", &TphtBackend::decide)
      .function("diagnosticsJson", &TphtBackend::diagnosticsJson)
      .function("stepHold", &TphtBackend::stepHold)
      .function("push", &TphtBackend::push)
      .function("pushActive", &TphtBackend::pushActive)
      .function("pushForce", &TphtBackend::pushForce)
      .function("setFrictionScale", &TphtBackend::setFrictionScale)
      .function("time", &TphtBackend::time)
      .function("timestep", &TphtBackend::timestep)
      .function("nq", &TphtBackend::nq)
      .function("nv", &TphtBackend::nv)
      .function("nu", &TphtBackend::nu)
      .function("qpos", &TphtBackend::qpos)
      .function("qvel", &TphtBackend::qvel)
      .function("torso", &TphtBackend::torso)
      .function("segments", &TphtBackend::segments)
      .function("ghostSegments", &TphtBackend::ghostSegments)
      .function("goldenJson", &TphtBackend::goldenJson);
}
