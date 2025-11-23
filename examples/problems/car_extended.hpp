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

  // Weights for each cost term (default = 1.0)
  double w_curv_rate = 1.0;     // u0 = κ̇
  double w_jerk = 1.0;          // u1 = jerk
  double w_centripetal_jerk = 1.0;
  double w_target_speed = 1.0;
  double w_lateral = 1.0;
  double w_centric_acc = 1.0;
  double w_terminal_state = 0.0;

  // Huber delta parameters
  double delta_speed = 1.0;      // for speed tracking
  double delta_lateral = 0.5;    // for lateral distance

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