#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <iostream>

#include "examples/traj_plan_cost.hpp"

namespace altro {
namespace examples {

// Helper: finite-difference gradient check
bool CheckGradient(const std::shared_ptr<problem::CostFunction>& cost,
                   const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                   double tol = 1e-6) {
  Eigen::VectorXd dx_analytic(x.size());
  Eigen::VectorXd du_analytic(u.size());
  cost->Gradient(x, u, dx_analytic, du_analytic);

  // Numerical gradient w.r.t x
  Eigen::VectorXd dx_num = Eigen::VectorXd::Zero(x.size());
  Eigen::VectorXd x_pert = x;
  double eps = 1e-6;
  for (int i = 0; i < x.size(); ++i) {
    x_pert(i) = x(i) + eps;
    double f_plus = cost->Evaluate(x_pert, u);
    x_pert(i) = x(i) - eps;
    double f_minus = cost->Evaluate(x_pert, u);
    dx_num(i) = (f_plus - f_minus) / (2 * eps);
    x_pert(i) = x(i);
  }

  // Numerical gradient w.r.t u
  Eigen::VectorXd du_num = Eigen::VectorXd::Zero(u.size());
  Eigen::VectorXd u_pert = u;
  for (int i = 0; i < u.size(); ++i) {
    u_pert(i) = u(i) + eps;
    double f_plus = cost->Evaluate(x, u_pert);
    u_pert(i) = u(i) - eps;
    double f_minus = cost->Evaluate(x, u_pert);
    du_num(i) = (f_plus - f_minus) / (2 * eps);
    u_pert(i) = u(i);
  }

  bool ok_x = (dx_analytic - dx_num).norm() <= tol * (1 + dx_num.norm());
  bool ok_u = (du_analytic - du_num).norm() <= tol * (1 + du_num.norm());

  if (!ok_x || !ok_u) {
    std::cout << "Gradient mismatch!\nAnalytic dx:\n" << dx_analytic.transpose()
              << "\nNumeric dx:\n" << dx_num.transpose()
              << "\nAnalytic du:\n" << du_analytic.transpose()
              << "\nNumeric du:\n" << du_num.transpose() << "\n";
  }
  return ok_x && ok_u;
}

// Helper: finite-difference Hessian check
bool CheckHessian(const std::shared_ptr<problem::CostFunction>& cost,
                  const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                  double tol = 1e-5) {
  Eigen::MatrixXd dxdx_ana(x.size(), x.size());
  Eigen::MatrixXd dxdu_ana(x.size(), u.size());
  Eigen::MatrixXd dudu_ana(u.size(), u.size());
  cost->Hessian(x, u, dxdx_ana, dxdu_ana, dudu_ana);

  // Numerical Hessian via gradient finite diff
  Eigen::MatrixXd dxdx_num = Eigen::MatrixXd::Zero(x.size(), x.size());
  Eigen::MatrixXd dxdu_num = Eigen::MatrixXd::Zero(x.size(), u.size());
  Eigen::MatrixXd dudu_num = Eigen::MatrixXd::Zero(u.size(), u.size());

  double eps = 1e-5;

  // d²f/dx²
  for (int i = 0; i < x.size(); ++i) {
    Eigen::VectorXd x_plus = x;
    x_plus(i) += eps;
    Eigen::VectorXd x_minus = x;
    x_minus(i) -= eps;

    Eigen::VectorXd dx_plus(x.size()), dx_minus(x.size());
    Eigen::VectorXd dummy_u(u.size());
    cost->Gradient(x_plus, u, dx_plus, dummy_u);
    cost->Gradient(x_minus, u, dx_minus, dummy_u);

    dxdx_num.col(i) = (dx_plus - dx_minus) / (2 * eps);
  }

  // d²f/du²
  for (int i = 0; i < u.size(); ++i) {
    Eigen::VectorXd u_plus = u;
    u_plus(i) += eps;
    Eigen::VectorXd u_minus = u;
    u_minus(i) -= eps;

    Eigen::VectorXd du_plus(u.size()), du_minus(u.size());
    Eigen::VectorXd dummy_x(x.size());
    cost->Gradient(x, u_plus, dummy_x, du_plus);
    cost->Gradient(x, u_minus, dummy_x, du_minus);

    dudu_num.col(i) = (du_plus - du_minus) / (2 * eps);
  }

  // d²f/dxdu
  for (int i = 0; i < x.size(); ++i) {
    Eigen::VectorXd x_plus = x;
    x_plus(i) += eps;
    Eigen::VectorXd x_minus = x;
    x_minus(i) -= eps;

    Eigen::VectorXd du_plus(u.size()), du_minus(u.size());
    Eigen::VectorXd dummy_x(x.size());
    cost->Gradient(x_plus, u, dummy_x, du_plus);
    cost->Gradient(x_minus, u, dummy_x, du_minus);

    dxdu_num.row(i) = (du_plus - du_minus).transpose() / (2 * eps);
  }

  bool ok_dxdx = (dxdx_ana - dxdx_num).norm() <= tol * (1 + dxdx_num.norm());
  bool ok_dudu = (dudu_ana - dudu_num).norm() <= tol * (1 + dudu_num.norm());
  bool ok_dxdu = (dxdu_ana - dxdu_num).norm() <= tol * (1 + dxdu_num.norm());

  if (!ok_dxdx || !ok_dudu || !ok_dxdu) {
    std::cout << "Hessian mismatch! Norms: dxdx=" << (dxdx_ana - dxdx_num).norm()
              << ", dxdu=" << (dxdu_ana - dxdu_num).norm()
              << ", dudu=" << (dudu_ana - dudu_num).norm() << "\n";
  }
  return ok_dxdx && ok_dudu && ok_dxdu;
}

// Common test state/control
Eigen::VectorXd MakeTestState() {
  Eigen::VectorXd x(6);
  x << 1.0, 2.0, 0.5, 0.1, 5.0, 1.0; // x, y, θ, κ, v, a
  return x;
}

Eigen::VectorXd MakeTestControl() {
  Eigen::VectorXd u(2);
  u << 0.02, 0.5; // κ̇, jerk
  return u;
}

// ---------- Test 1: CentripetalAccelerationCost ----------
TEST(TrajPlanCostTest, CentripetalAccelerationCost) {
  auto cost = std::make_shared<CentripetalAccelerationCost>(2.0, false);
  auto x = MakeTestState();
  auto u = MakeTestControl();

  // Evaluate
  double val = cost->Evaluate(x, u);
  double expected = 0.5 * 2.0 * std::pow(5.0 * 5.0 * 0.1, 2); // 0.5*w*(v²κ)²
  EXPECT_NEAR(val, expected, 1e-9);

  // Gradient & Hessian
  EXPECT_TRUE(CheckGradient(cost, x, u));
  EXPECT_TRUE(CheckHessian(cost, x, u));
}

// ---------- Test 2: CentripetalJerkCost ----------
TEST(TrajPlanCostTest, CentripetalJerkCost) {
  auto cost = std::make_shared<CentripetalJerkCost>(1.5, false);
  auto x = MakeTestState();
  auto u = MakeTestControl();

  double v = x(4), a = x(5), kappa = x(3), kappa_dot = u(0);
  double term = 2.0 * v * a * kappa + v * v * kappa_dot;
  double expected = 0.5 * 1.5 * term * term;
  double val = cost->Evaluate(x, u);
  EXPECT_NEAR(val, expected, 1e-9);

  EXPECT_TRUE(CheckGradient(cost, x, u));
  EXPECT_TRUE(CheckHessian(cost, x, u));
}

// ---------- Test 3: CurvatureRateCost (Quadratic) ----------
TEST(TrajPlanCostTest, CurvatureRateCost) {
  auto cost = std::make_shared<CurvatureRateCost>(3.0, false);
  auto x = MakeTestState();
  auto u = MakeTestControl();

  // Only u(0) penalized: 0.5 * w * u0^2
  double expected = 0.5 * 3.0 * u(0) * u(0);
  double val = cost->Evaluate(x, u);
  EXPECT_NEAR(val, expected, 1e-9);

  EXPECT_TRUE(CheckGradient(cost, x, u));
  EXPECT_TRUE(CheckHessian(cost, x, u));
}

// ---------- Test 4: LinearJerkCost (Quadratic) ----------
TEST(TrajPlanCostTest, LinearJerkCost) {
  auto cost = std::make_shared<LinearJerkCost>(4.0, false);
  auto x = MakeTestState();
  auto u = MakeTestControl();

  double expected = 0.5 * 4.0 * u(1) * u(1);
  double val = cost->Evaluate(x, u);
  EXPECT_NEAR(val, expected, 1e-9);

  EXPECT_TRUE(CheckGradient(cost, x, u));
  EXPECT_TRUE(CheckHessian(cost, x, u));
}

// ---------- Test 5: LateralDistanceHuberCost ----------
TEST(TrajPlanCostTest, LateralDistanceHuberCost) {
  Eigen::Vector2d proj_pos(1.1, 2.2);
  auto cost = std::make_shared<LateralDistanceHuberCost>(proj_pos, 10.0, 0.5);
  auto x = MakeTestState(); // x=1.0, y=2.0
  auto u = MakeTestControl();

  double dist = std::sqrt((1.0 - 1.1)*(1.0 - 1.1) + (2.0 - 2.2)*(2.0 - 2.2)); // ≈0.2236
  double expected = 10.0 * HuberLoss(dist, 0.5); // quadratic region
  double val = cost->Evaluate(x, u);
  EXPECT_NEAR(val, expected, 1e-9);

  EXPECT_TRUE(CheckGradient(cost, x, u));
  EXPECT_TRUE(CheckHessian(cost, x, u));
}

// ---------- Test 6: TargetSpeedHuberCost ----------
TEST(TrajPlanCostTest, TargetSpeedHuberCost) {
  const double weight = 5.0;
  const double v_target = 6.0;
  const double delta = 1.0;
  auto cost = std::make_shared<TargetSpeedHuberCost>(weight, v_target, delta, false);
  auto x = MakeTestState(); // v = 5.0
  auto u = MakeTestControl();

  double error = x(4) - v_target; // 5.0 - 6.0 = -1.0
  double val = cost->Evaluate(x, u);
  double expected_val = weight * HuberLoss(error, delta); // = 5 * 0.5 = 2.5
  EXPECT_NEAR(val, expected_val, 1e-9);

  // Gradient should still be fine at boundary (subgradient is continuous)
  EXPECT_TRUE(CheckGradient(cost, x, u));

  // ❗ Hessian is discontinuous at |error| == delta.
  // So we test Hessian at a point strictly inside quadratic region.
  Eigen::VectorXd x_inside = x;
  x_inside(4) = v_target + 0.5 * delta; // error = +0.5 < delta
  EXPECT_TRUE(CheckHessian(cost, x_inside, u));

  // Also test in linear region
  Eigen::VectorXd x_outside = x;
  x_outside(4) = v_target + 2.0 * delta; // error = +2.0 > delta
  EXPECT_TRUE(CheckHessian(cost, x_outside, u));
}

}  // namespace examples
}  // namespace altro