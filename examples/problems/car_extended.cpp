// Copyright [2025] Your Name or Org

#include <memory>
#include "examples/problems/car_extended.hpp"

namespace altro {
namespace problems {

CarExtendedProblem::CarExtendedProblem() {
  // Default initial/final states
  x0 << 0, 0, 0, 0, 0.0, 0;   // [x, y, θ, κ, v, a]
  xf << 0, 0, 0, 0, 0.0, 0;
  u0 << 0, 0;
}

void CarExtendedProblem::SetScenario(Scenario scenario) {
  scenario_ = scenario;
  std::vector<Eigen::Vector4d> traj;

  switch (scenario) {
    case kLaneChange:
      break;
    case kObstacleAvoidance:
      break;
    case kQuarterTurn:
      break;
    case kUturn:
      break;
    case kGtest:
      N = 100;
      tf = 10.0;
      x0 << 0, 0, 0, 0, 0.0, 0;   // [x, y, θ, κ, v, a]
      xf << 20, 1, 0, 0, 10.0, 0;
      u0 << 0.05, 0.1;

      traj.resize(N+1);
      for (int k = 0; k <= N; ++k) {
        double x_ref = xf(0) * static_cast<double>(k) / N;  // 0 → 20
        double y_ref = 0.0;                                 // fixed!
        double theta_ref = 0.0;                             // along x-axis
        double v_ref = xf(4);                               // 10.0

        traj[k] = Eigen::Vector4d(x_ref, y_ref, theta_ref, v_ref);
      }
      ref_line_ = std::make_shared<altro::examples::ReferenceLine>(std::move(traj));

      break;
    default:
      throw std::invalid_argument("Unknown scenario");
  }

  // if (scenario == kLaneChange) {
  //   N = 100;
  //   tf = 4.0;
  //   x0 << 0, 0, 0, 0, 5.0, 0;
  //   xf << 30, 0, 0, 0, 5.0, 0;
  //   obstacles.clear();
  // } else if (scenario == kObstacleAvoidance) {
  //   N = 150;
  //   tf = 6.0;
  //   x0 << 0, 0, 0, 0, 5.0, 0;
  //   xf << 40, 2.0, 0, 0, 5.0, 0;
  //   obstacles = {
  //       Eigen::Vector3d(15, 0.5, 1.0),
  //       Eigen::Vector3d(25, -0.5, 1.0),
  //       Eigen::Vector3d(35, 0.0, 1.2)
  //   };
  // }

}

altro::problem::Problem CarExtendedProblem::MakeProblem(bool add_constraints) {
  const double h = GetTimeStep();
  altro::problem::Problem prob(N);

  // === Dynamics ===
  for (int k = 0; k < N; ++k) {
    prob.SetDynamics(std::make_shared<ModelDis<>>(model), k);
  }

  // === Cost Functions (per stage) ===
  for (int k = 0; k < N; ++k) {
    std::vector<std::shared_ptr<altro::problem::CostFunction>> costs;

    // 1. Curvature rate penalty (u0)
    if (w_curv_rate > 0) {
      costs.push_back(
          std::make_shared<altro::examples::CurvatureRateCost>(w_curv_rate * h, false));
    }

    // 2. Jerk penalty (u1)
    if (w_jerk > 0) {
      costs.push_back(
          std::make_shared<altro::examples::LinearJerkCost>(w_jerk * h, false));
    }

    // 3. Centripetal jerk penalty
    if (w_centripetal_jerk > 0) {
      costs.push_back(
          std::make_shared<altro::examples::CentripetalJerkCost>(w_centripetal_jerk * h, false));
    }

    // 4. Target speed (Huber)
    // todo: add reference speed profile support
    if (w_target_speed > 0) {
      if (scenario_ == kGtest) {
        costs.push_back(std::make_shared<altro::examples::TargetSpeedHuberCost>(
            w_target_speed * h, xf(4), delta_speed, false));
      }
    }

    // 5. Lateral distance to reference line (y=0) — Huber
    // todo: add reference line profile support
    if (w_lateral > 0) {
      if (scenario_ == kGtest) {
        Eigen::Vector2d ref_point(xf(0) * static_cast<double>(k) / N, 0.0);  // moving target
        costs.push_back(std::make_shared<altro::examples::LateralDistanceHuberCost>(
            ref_point, w_lateral * h, delta_lateral, false));
      }
    }

    // tracking cost
    if (w_target_speed > 0 || w_lateral > 0)
    {
      auto tracking_cost = std::make_shared<altro::examples::ReferenceTrackingCost>(
          ref_line_,
          w_lateral * h,
          w_target_speed * h,
          delta_lateral,
          delta_speed,
          false  // terminal flag
      );
      costs.push_back(tracking_cost);
    }

    if (w_centric_acc > 0) {
      costs.push_back(
          std::make_shared<altro::examples::CentripetalAccelerationCost>(w_centric_acc * h, false));
    }

    // Combine all stage costs into a SumCost (if needed)
    if (!costs.empty()) {
      auto sum_cost = std::make_shared<altro::examples::SumCost>(costs);
      prob.SetCostFunction(sum_cost, k);
    }
  }

  // === Terminal cost ===
  if (w_terminal_state > 0) {
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(NStates, NStates) * (w_terminal_state * 10.0);
    Eigen::VectorXd qf = -Qf * xf;  // linear term
    double cf = 0.5 * xf.transpose() * Qf * xf;
    auto term_cost = std::make_shared<altro::examples::QuadraticCost>(
        Qf,                                    // Q
        Eigen::MatrixXd::Zero(NControls, NControls),  // R (nu×nu)
        Eigen::MatrixXd::Zero(NStates, NControls),    // H (nx×nu)
        qf,                                    // q
        Eigen::VectorXd::Zero(NControls),      // r
        cf,                                   // c (constant, optional)
        true                                   // terminal flag
    );
    prob.SetCostFunction(term_cost, N);
  }

  // === Constraints ===
  if (add_constraints) {
    // Control bounds: [v_dot_min, jerk_min] to [v_dot_max, jerk_max]
    std::vector<double> lb = {-3.0, -5.0};
    std::vector<double> ub = {+3.0, +5.0};
    for (int k = 0; k < N; ++k) {
      prob.SetConstraint(
          std::make_shared<altro::examples::ControlBound>(lb, ub), k);
    }

    // Terminal state constraint (soft via cost, or hard if desired)
    // prob.SetConstraint(
    //     std::make_shared<altro::examples::GoalConstraint>(xf), N);

    // Obstacle constraints (only in obstacle scenario)
    // if (scenario_ == kObstacleAvoidance && !obstacles.empty()) {
    //   for (int k = 1; k < N; ++k) {
    //     auto obs_constraint = std::make_shared<altro::examples::CircleConstraint>();
    //     for (const auto& obs : obstacles) {
    //       obs_constraint->AddObstacle(obs(0), obs(1), obs(2));
    //     }
    //     prob.SetConstraint(obs_constraint, k);
    //   }
    // }
  }

  // === Initial state ===
  prob.SetInitialState(x0);

  return prob;
}

}  // namespace problems
}  // namespace altro