// Copyright [2025] Your Name or Org

#pragma once

#include <memory>
#include <vector>

#include "altro/eigentypes.hpp"
#include "altro/problem/problem.hpp"
#include "altro/common/trajectory.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/problem/integration.hpp"
#include "altro/problem/discretized_model.hpp"
#include "altro/augmented_lagrangian/al_solver.hpp"

#include "examples/basic_constraints.hpp"
#include "examples/obstacle_constraints.hpp"
#include "examples/traj_plan_cost.hpp"
#include "examples/car_kin.hpp"

// Forward declare custom costs (you must have these defined elsewhere)
class CurvatureRateCost;
class LinearJerkCost;
class CentripetalJerkCost;
class TargetSpeedHuberCost;
class LateralDistanceHuberCost;
class QuadraticCost;

namespace altro {
namespace problems {

class CarExtendedProblem {
 public:
  static constexpr int NStates = 6;
  static constexpr int NControls = 2;

  enum Scenario { kLaneChange, kObstacleAvoidance, kQuarterTurn, kUturn, kGtest};

  using TrajType = altro::Trajectory<NStates, NControls>;
  using ModelCon = altro::examples::CarExtended;
  template <class IntegratorSpecific = altro::problem::RungeKutta4Fast<NStates, NControls>>
  using ModelDis = altro::problem::DiscretizedModel<ModelCon, IntegratorSpecific>;
  altro::examples::CarExtended model = ModelCon();

  CarExtendedProblem();

  // ==============================
  // 1. Original weight intent (designer’s subjective importance)
  // ==============================
  double w_curv_rate_orin     = 1.0;   // curvature change rate
  double w_jerk_orin          = 1.0;   // longitudinal jerk
  double w_centripetal_jerk_orin = 1.0;
  double w_target_speed_orin  = 1.0;   // speed tracking
  double w_lateral_orin       = 1.0;   // lateral offset
  double w_centric_acc_orin   = 1.0;   // centripetal acceleration
  double w_terminal_state_orin = 1.0;  // terminal accuracy
  const double C = 5.0;                // cost value according to max cost term

  // State constraints parameters
  double a_min = -5.0;
  double a_max = 2.0;
  double v_min = 0.0;
  double v_max = 40;  // m/s 144km/h
  double kappa_min = -0.25;
  double kappa_max = 0.25;
  double centric_acc_max = 7.0;
  double centric_jerk_max = 100;
  double heading_offset_max = M_PI / 3.0;  // 60 degrees

  // Control constraints parameters
  double kappa_dot_min = -0.6;   // 1/(m·s)
  double kappa_dot_max =  0.6;
  double jerk_min = -5.0;   // m/s³
  double jerk_max =  5.0;

  // Huber delta parameters
  double delta_speed = 1.0;      // for speed tracking
  double delta_lateral = 0.5;    // for lateral distance

  double terminal_pos_tol   = 0.1;       // m
  double terminal_yaw_tol   = M_PI / 36.0; // 5 degrees ≈ 0.087 rad

  // Weights for each cost term (default = 1.0)
  double w_curv_rate = w_curv_rate_orin * C / (kappa_dot_max * kappa_dot_max);     // u0 = κ̇
  double w_jerk = w_jerk_orin * C / (jerk_max * jerk_max);                         // u1 = jerk
  double w_centripetal_jerk = w_centripetal_jerk_orin * C / (centric_jerk_max * centric_jerk_max);
  double w_target_speed = w_target_speed_orin * C / (delta_speed * delta_speed);
  double w_lateral = w_lateral_orin * C / (delta_lateral * delta_lateral);
  double w_centric_acc = w_centric_acc_orin * C / (centric_acc_max * centric_acc_max);
  double w_terminal_pos = w_terminal_state_orin * C / (terminal_pos_tol * terminal_pos_tol);
  double w_terminal_yaw = w_terminal_state_orin * C / (terminal_yaw_tol * terminal_yaw_tol);
  double w_terminal_state = w_terminal_pos;

  // Problem parameters
  int N = 100;
  double tf = 10.0;
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(NStates);
  Eigen::VectorXd xf = Eigen::VectorXd::Zero(NStates);
  Eigen::Vector2d u0 = Eigen::Vector2d::Zero();

  void SetScenario(Scenario scenario);
  Scenario GetScenario() const { return scenario_; }
  
  float GetTimeStep() const { return static_cast<float>(tf) / N; }

  altro::problem::Problem MakeProblem(bool add_constraints = true);

  std::shared_ptr<const altro::examples::ReferenceLine> GetReferenceLine() const {
    return std::const_pointer_cast<const altro::examples::ReferenceLine>(ref_line_);
}

  template <int n_size = NStates, int m_size = NControls>
  TrajType InitialTrajectory();

  template <int n_size = NStates, int m_size = NControls>
  altro::ilqr::iLQR<n_size, m_size> MakeSolver(bool alcost = true);

  template <int n_size = NStates, int m_size = NControls>
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<n_size, m_size> MakeALSolver();

 private:
  Scenario scenario_ = kQuarterTurn;
  std::shared_ptr<altro::examples::ReferenceLine> ref_line_;
  std::shared_ptr<altro::examples::ReferenceLineProjector> projector_;
};

// Implementations
template <int n_size, int m_size>
auto CarExtendedProblem::InitialTrajectory() -> TrajType {
  TrajType Z(NStates, NControls, N);
  for (int k = 0; k < N; ++k) {
    Z.Control(k) = u0;
  }
  Z.SetUniformStep(GetTimeStep());
  return Z;
}

template <int n_size, int m_size>
auto CarExtendedProblem::MakeSolver(bool alcost) -> altro::ilqr::iLQR<n_size, m_size> {
  auto prob = MakeProblem();
  if (alcost) {
    prob = altro::augmented_lagrangian::BuildAugLagProblem<n_size, m_size>(prob);
  }
  altro::ilqr::iLQR<n_size, m_size> solver(prob);
  auto traj = std::make_shared<TrajType>(InitialTrajectory());
  solver.SetTrajectory(traj);
  solver.Rollout();
  return solver;
}

template <int n_size, int m_size>
auto CarExtendedProblem::MakeALSolver()
    -> altro::augmented_lagrangian::AugmentedLagrangianiLQR<n_size, m_size> {
  auto prob = MakeProblem(true);
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<n_size, m_size> solver(prob);
  solver.SetTrajectory(std::make_shared<TrajType>(InitialTrajectory()));
  solver.GetiLQRSolver().Rollout();
  return solver;
}

}  // namespace problems
}  // namespace altro