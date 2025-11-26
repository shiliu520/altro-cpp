// Copyright [2025] Your Name or Org

#include <memory>
#include "examples/problems/car_extended.hpp"

// 辅助函数：带过渡的半宽计算
double ComputeHalfWidthWithTaper(
    double x,
    double x_start_narrow,
    double x_end_narrow,
    double full_width = 1.5,
    double intrusion = 0.8,
    double transition_len = 1.0) {

    double min_width = full_width - intrusion; // e.g., 0.7
    double x_enter = x_start_narrow - transition_len;
    double x_exit  = x_end_narrow   + transition_len;

    if (x <= x_enter) return full_width;
    if (x >= x_exit)  return full_width;

    if (x < x_start_narrow) {
        // 进入过渡区: [x_enter, x_start_narrow] → full → min
        return full_width - intrusion * (x - x_enter) / transition_len;
    } else if (x <= x_end_narrow) {
        // 完全变窄区
        return min_width;
    } else {
        // 退出过渡区: [x_end_narrow, x_exit] → min → full
        return min_width + intrusion * (x - x_end_narrow) / transition_len;
    }
}

// 生成左右边界 ReferenceLine
void GenerateRoadBoundaries(
    const std::vector<Eigen::Vector4d>& centerline,
    std::shared_ptr<altro::examples::ReferenceLine>& left_boundary,
    std::shared_ptr<altro::examples::ReferenceLine>& right_boundary) {

    // Step 1: 初始粗略生成边界点（和之前一样）
    std::vector<Eigen::Vector2d> left_pts_coarse, right_pts_coarse;
    std::vector<double> vels_coarse;
    std::vector<double> xs; // 保存 x 坐标用于判断区域

    for (const auto& ref : centerline) {
        double x = ref(0), y = ref(1), theta = ref(2), v = ref(3);
        Eigen::Vector2d n_left(-std::sin(theta), std::cos(theta));
        double left_hw  = ComputeHalfWidthWithTaper(x, 5.0, 10.0);
        double right_hw = ComputeHalfWidthWithTaper(x, 20.0, 25.0);
        Eigen::Vector2d center(x, y);
        left_pts_coarse.push_back(center + left_hw * n_left);
        right_pts_coarse.push_back(center - right_hw * n_left);
        vels_coarse.push_back(v);
        xs.push_back(x);
    }

    // Step 2: 自适应重采样边界轨迹
    auto resampleBoundary = [&](const std::vector<Eigen::Vector2d>& pts,
                                const std::vector<double>& xs_input,
                                const std::vector<double>& vels_input,
                                double base_density = 0.1) -> std::vector<Eigen::Vector4d> {
        std::vector<Eigen::Vector4d> refined;

        for (size_t i = 0; i < pts.size() - 1; ++i) {
            Eigen::Vector2d p0 = pts[i];
            Eigen::Vector2d p1 = pts[i + 1];
            double x0 = xs_input[i];
            double x1 = xs_input[i + 1];

            // 判断这段是否在“变窄过渡区”
            bool in_transition = false;
            // 检查左/右是否在过渡区（简化：只要半宽变化快）
            double hw0_left = ComputeHalfWidthWithTaper(x0, 5.0, 10.0);
            double hw1_left = ComputeHalfWidthWithTaper(x1, 5.0, 10.0);
            double hw0_right = ComputeHalfWidthWithTaper(x0, 20.0, 25.0);
            double hw1_right = ComputeHalfWidthWithTaper(x1, 20.0, 25.0);

            if (std::abs(hw1_left - hw0_left) > 1e-3 ||
                std::abs(hw1_right - hw0_right) > 1e-3) {
                in_transition = true;
            }

            double seg_len = (p1 - p0).norm();
            int num_seg = in_transition ?
                std::max(2, static_cast<int>(seg_len / (base_density * 0.2))) : // 加密5倍
                std::max(1, static_cast<int>(seg_len / base_density));

            for (int j = 0; j <= num_seg; ++j) {
                double alpha = static_cast<double>(j) / num_seg;
                Eigen::Vector2d p_interp = p0 + alpha * (p1 - p0);
                double v_interp = vels_input[i] + alpha * (vels_input[i+1] - vels_input[i]);

                // 先存位置，后面统一算 heading
                refined.emplace_back(p_interp.x(), p_interp.y(), 0.0, v_interp);
            }
        }
        return refined;
    };

    auto left_traj_refined = resampleBoundary(left_pts_coarse, xs, vels_coarse);
    auto right_traj_refined = resampleBoundary(right_pts_coarse, xs, vels_coarse);

    // Step 3: 对加密后的轨迹计算 heading（数值微分）
    auto computeHeadingsInPlace = [](std::vector<Eigen::Vector4d>& traj) {
        size_t N = traj.size();
        if (N <= 1) return;

        // 首点
        double dx0 = traj[1](0) - traj[0](0);
        double dy0 = traj[1](1) - traj[0](1);
        traj[0](2) = std::atan2(dy0, dx0);

        // 中间点
        for (size_t k = 1; k < N - 1; ++k) {
            double dx = traj[k+1](0) - traj[k-1](0);
            double dy = traj[k+1](1) - traj[k-1](1);
            traj[k](2) = std::atan2(dy, dx);
        }

        // 末点
        double dxN = traj[N-1](0) - traj[N-2](0);
        double dyN = traj[N-1](1) - traj[N-2](1);
        traj[N-1](2) = std::atan2(dyN, dxN);
    };

    computeHeadingsInPlace(left_traj_refined);
    computeHeadingsInPlace(right_traj_refined);

    left_boundary  = std::make_shared<altro::examples::ReferenceLine>(std::move(left_traj_refined));
    right_boundary = std::make_shared<altro::examples::ReferenceLine>(std::move(right_traj_refined));
}

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
    case kQuarterTurn: {
      N = 100;
      tf = 10.0;  // total 10 seconds

      // Initial state: start from rest at origin, heading along x-axis
      x0 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;  // [x, y, θ, κ, v, a]
      u0 << 0.0, 0.0;

      // Parameters
      const double R = 31.83;        // turning radius (~1/0.0314)
      const double v_target = 10.0;  // 36 km/h
      const double t_start_turn = 3.0;
      const double t_end_turn = 8.0;
      const double kappa_turn = 1.0 / R;

      // Compute straight segment length at start of turn
      double L = 0.0;
      if (t_start_turn <= 2.0) {
        L = v_target * t_start_turn;  // still accelerating
      } else {
        L = v_target * 2.0
            + v_target * (t_start_turn - 2.0);  // accelerate for 2s, then constant speed
      }
      // Final state must match the end of the reference trajectory!
      xf << L + R, -R, -M_PI_2, 0.0, v_target, 0.0;

      // Build reference trajectory: piecewise (straight -> arc -> straight)
      traj.resize(N + 1);

      for (int k = 0; k <= N; ++k) {
        double t = tf * static_cast<double>(k) / N;  // t in [0, tf]

        Eigen::Vector4d ref;  // [x_ref, y_ref, theta_ref, v_ref]

        if (t < t_start_turn) {
          // Before turn: straight along x-axis
          double s = (t <= 2.0) ? v_target * t : v_target * 2.0 + v_target * (t - 2.0);
          ref << s, 0.0, 0.0, v_target;

        } else if (t <= t_end_turn) {
          // During turn: circular arc, center at (L, R)
          double dt = t - t_start_turn;
          double phi = kappa_turn * v_target * dt;  // positive angle swept

          double x_arc = L + R * std::sin(phi);
          double y_arc = -R * (1.0 - std::cos(phi));  // = R*(cos(phi) - 1), negative
          double theta_ref = -phi;                    // heading turns clockwise

          ref << x_arc, y_arc, theta_ref, v_target;

        } else {
          // After turn: straight along -y direction
          double dt_after = t - t_end_turn;
          double x_final = L + R;
          double y_final = -R - v_target * dt_after;
          ref << x_final, y_final, -M_PI_2, v_target;
        }

        traj[k] = ref;
      }

      ref_line_ = std::make_shared<altro::examples::ReferenceLine>(std::move(traj));
      projector_ = std::make_shared<altro::examples::ReferenceLineProjector>(ref_line_);

      std::shared_ptr<altro::examples::ReferenceLine> left_ref, right_ref;
      GenerateRoadBoundaries(ref_line_->GetTrajectory(), left_ref, right_ref);

      // 创建投影器（用于 PathBoundConstraint）
      left_projector_  = std::make_shared<altro::examples::ReferenceLineProjector>(left_ref);
      right_projector_ = std::make_shared<altro::examples::ReferenceLineProjector>(right_ref);

      // std::cout << "create left/right boundary projectors for quarter turn scenario.\n";

      break;
    }
    case kUturn:
      break;
    case kGtest:
      N = 100;
      tf = 10.0;
      x0 << 0, 0, 0, 0, 0.1, 0;   // [x, y, θ, κ, v, a], v(0) = 0.1 to avoid zero initial speed, which can generate constraint derivative
      xf << 20, 1, 0, 0, 10.0, 0;
      u0 << 0.0, 0.1;             // u0(0) > 0 can cause kappa to increase, even violate constraints, which can generate constraint cost

      traj.resize(N+1);
      for (int k = 0; k <= N; ++k) {
        double x_ref = xf(0) * static_cast<double>(k) / N;  // 0 → 20
        double y_ref = 0.0;                                 // fixed!
        double theta_ref = 0.0;                             // along x-axis
        double v_ref = xf(4);                               // 10.0

        traj[k] = Eigen::Vector4d(x_ref, y_ref, theta_ref, v_ref);
      }
      ref_line_ = std::make_shared<altro::examples::ReferenceLine>(std::move(traj));
      projector_ = std::make_shared<altro::examples::ReferenceLineProjector>(ref_line_);

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
    // if (w_target_speed > 0) {
    //   if (scenario_ == kGtest) {
    //     costs.push_back(std::make_shared<altro::examples::TargetSpeedHuberCost>(
    //         w_target_speed * h, xf(4), delta_speed, false));
    //   }
    // }

    // 5. Lateral distance to reference line (y=0) — Huber
    // todo: add reference line profile support
    // if (w_lateral > 0) {
    //   if (scenario_ == kGtest) {
    //     Eigen::Vector2d ref_point(xf(0) * static_cast<double>(k) / N, 0.0);  // moving target
    //     costs.push_back(std::make_shared<altro::examples::LateralDistanceHuberCost>(
    //         ref_point, w_lateral * h, delta_lateral, false));
    //   }
    // }

    // tracking cost
    if (w_target_speed > 0 || w_lateral > 0)
    {
      auto tracking_cost = std::make_shared<altro::examples::ReferenceTrackingCost>(
          projector_,
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
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(NStates, NStates) * (w_terminal_state * 1.0);
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
    std::vector<double> lb_control = {kappa_dot_min, jerk_min};
    std::vector<double> ub_control = {kappa_dot_max, jerk_max};
    std::vector<double> lb_state = {-ALTRO_INFINITY, -ALTRO_INFINITY, -ALTRO_INFINITY, kappa_min, v_min, a_min};
    std::vector<double> ub_state = {ALTRO_INFINITY, ALTRO_INFINITY, ALTRO_INFINITY, kappa_max, v_max, a_max};
    for (int k = 0; k < N; ++k) {
      prob.SetConstraint(
          std::make_shared<altro::examples::ControlBound>(lb_control, ub_control), k);
      prob.SetConstraint(
          std::make_shared<altro::examples::StateBound>(lb_state, ub_state), k);
      prob.SetConstraint(
          std::make_shared<altro::examples::CentripetalAccelerationConstraint>(centric_acc_max), k);
      prob.SetConstraint(
          std::make_shared<altro::examples::CentripetalJerkConstraint>(centric_jerk_max), k);
      prob.SetConstraint(
          std::make_shared<altro::examples::HeadingTrackingConstraint>(projector_, heading_offset_max), k);
      prob.SetConstraint(
          std::make_shared<altro::examples::SpeedTrackingConstraint>(projector_), k);
      if (GetScenario() != kGtest) {
        prob.SetConstraint(
            std::make_shared<altro::examples::PathBoundConstraint>(left_projector_, right_projector_, model), k);
      }

    }
    // prob.SetConstraint(std::make_shared<examples::GoalConstraint>(xf), N);

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