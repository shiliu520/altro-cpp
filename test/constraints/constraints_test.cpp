// Copyright [2021] Optimus Ride Inc.

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include "altro/constraints/constraint.hpp"
#include "altro/constraints/constraint_values.hpp"
#include "altro/eigentypes.hpp"
#include "altro/problem/problem.hpp"
#include "altro/utils/assert.hpp"
#include "altro/utils/derivative_checker.hpp"
#include "examples/basic_constraints.hpp"
#include "examples/obstacle_constraints.hpp"

Eigen::MatrixXd ComputeNumericalJacobian(
    const std::function<void(const Eigen::VectorXd&, const Eigen::VectorXd&, Eigen::Ref<Eigen::VectorXd>)>& eval_func,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u,
    const int32_t n_constraints,
    double eps = 1e-6) {

    Eigen::VectorXd c0(n_constraints);
    eval_func(x, u, c0);
    int nc = c0.size();
    int nx = x.size();
    int nu = u.size();

    Eigen::MatrixXd jac(nc, nx + nu);

    // Perturb states
    for (int i = 0; i < nx; ++i) {
        Eigen::VectorXd x_perturb = x;
        x_perturb(i) += eps;
        Eigen::VectorXd c1(n_constraints);
        eval_func(x_perturb, u, c1);
        jac.col(i) = (c1 - c0) / eps;
    }

    // Perturb controls
    for (int i = 0; i < nu; ++i) {
        Eigen::VectorXd u_perturb = u;
        u_perturb(i) += eps;
        Eigen::VectorXd c1(n_constraints);
        eval_func(x, u_perturb, c1);   // ← 关键：用原始 x！
        jac.col(nx + i) = (c1 - c0) / eps;
    }

    return jac;
}

namespace altro {

TEST(BasicConstraints, ControlBoundConstructor) {
  int m = 3;
  double inf = std::numeric_limits<double>::infinity();
  examples::ControlBound bnd(m);
  EXPECT_EQ(bnd.OutputDimension(), 0);
  std::vector<double> lb = {-inf, -2, -3};
  bnd.SetLowerBound(lb);
  EXPECT_EQ(bnd.OutputDimension(), 2);

  lb = {-inf, 0, -inf};
  bnd.SetLowerBound(lb);
  EXPECT_EQ(bnd.OutputDimension(), 1);

  std::vector<double> ub = {inf, inf, inf};
  bnd.SetUpperBound(ub);
  EXPECT_EQ(bnd.OutputDimension(), 1);

  ub = {1, 2, 3};
  bnd.SetUpperBound(ub);
  EXPECT_EQ(bnd.OutputDimension(), 4);

  // Test moving bounds
  bnd.SetUpperBound(std::move(ub));
  bnd.SetLowerBound(std::move(lb));
  EXPECT_EQ(ub.size(), 0);  // NOLINT
  EXPECT_EQ(lb.size(), 0);  // NOLINT
}

TEST(BasicConstraints, GoalConstructor) {
  Eigen::Vector4d xf(1.0, 2.0, 3.0, 4.0);
  examples::GoalConstraint goal(xf);
  EXPECT_EQ(goal.OutputDimension(), 4);

  VectorXd xf2(xf);
  examples::GoalConstraint goal2(xf2);
  EXPECT_EQ(goal2.OutputDimension(), 4);
}

TEST(BasicConstraints, GoalConstraint) {
  Eigen::Vector4d xf(1.0, 2.0, 3.0, 4.0);
  examples::GoalConstraint goal(xf);
  VectorXd c(goal.OutputDimension());
  Eigen::Vector4d x(1, 2, 3, 4);
  Eigen::Vector3d u(-1, -2, -3);
  goal.Evaluate(x, u, c);
  EXPECT_TRUE(c.isApprox(Eigen::Vector4d::Zero()));
  goal.Evaluate(2 * x, u, c);
  EXPECT_TRUE(c.isApprox(x));
  VectorXd x_bad = VectorXd::Constant(5, 2.0);
  if (utils::AssertionsActive()) {
    EXPECT_DEATH(goal.Evaluate(x_bad, u, c), "Assertion.*rows().*failed");
  }
}

TEST(CircleConstraint, Constructor) {
  examples::CircleConstraint obs;
  obs.AddObstacle(1.0, 2.0, 0.25);
  EXPECT_EQ(obs.OutputDimension(), 1);
  obs.AddObstacle(2.0, 4.0, 0.5);
  EXPECT_EQ(obs.OutputDimension(), 2);
}

TEST(CircleConstraint, Evaluate) {
  examples::CircleConstraint obs;
  Eigen::Vector2d p1(1.0, 2.0);
  Eigen::Vector2d p2(2.0, 4.0);
  obs.AddObstacle(p1(0), p1(1), 0.25);
  obs.AddObstacle(p2(0), p2(1), 0.5);

  const Eigen::Vector2d x(0.5, 1.5);
  const Eigen::Vector2d u(-0.25, 0.25);
  Eigen::Vector2d c = Eigen::Vector2d::Zero();
  obs.Evaluate(x, u, c);
  Eigen::Vector2d d1 = x - p1; 
  Eigen::Vector2d d2 = x - p2; 
  Eigen::Vector2d c_expected(0.25 * 0.25 - d1.squaredNorm(), 0.5 * 0.5 - d2.squaredNorm());
  EXPECT_TRUE(c.isApprox(c_expected));
}

TEST(CircleConstraint, Jacobian) {
  examples::CircleConstraint obs;
  Eigen::Vector2d p1(1.0, 2.0);
  Eigen::Vector2d p2(2.0, 4.0);
  obs.AddObstacle(p1(0), p1(1), 0.25);
  obs.AddObstacle(p2(0), p2(1), 0.5);

  const Eigen::Vector2d x(0.5, 1.5);
  const Eigen::Vector2d u(-0.25, 0.25);
  MatrixNxMd<2, 2> jac = Eigen::Matrix2d::Zero();
  obs.Jacobian(x, u, jac);
  Eigen::Vector2d d1 = x - p1; 
  Eigen::Vector2d d2 = x - p2; 
  MatrixXd jac_expected(2,2);
  jac_expected << -2 * d1(0), -2 * d1(1), -2 * d2(0), -2 * d2(1);

  auto eval = [&](auto x_) {
    VectorXd c_(2);
    obs.Evaluate(x_, u, c_);
    return c_;
  };
  VectorXd x2 = x;
  MatrixXd jac_fd = utils::FiniteDiffJacobian<-1, -1>(eval, x2);

  EXPECT_TRUE(jac.isApprox(jac_expected));
  EXPECT_TRUE(jac.isApprox(jac_fd, 1e-4));
}

class ConstraintValueTest : public ::testing::Test {
 protected:
  static constexpr int n_static = 4;
  static constexpr int m_static = 2;
  int n;
  int m;
  constraints::ConstraintPtr<constraints::Equality> goal;
  constraints::ConstraintPtr<constraints::Inequality> ubnd;

  void SetUp() override {
    n = n_static;
    m = m_static;
    Eigen::Vector4d xf(1.0, 2.0, 3.0, 4.0);
    goal = std::make_shared<examples::GoalConstraint>(xf);

    std::vector<double> lb = {-2, -3};
    std::vector<double> ub = {2, 3};
    ubnd = std::make_shared<examples::ControlBound>(lb, ub);
  }
};

TEST_F(ConstraintValueTest, Constructor) {
  constraints::ConstraintValues<n_static, m_static, constraints::Equality> conval(n, m, goal);
  EXPECT_EQ(conval.StateDimension(), n);
  EXPECT_EQ(conval.ControlDimension(), m);
}

TEST_F(ConstraintValueTest, ConstraintInterface) {
  constraints::ConstraintValues<n_static, m_static, constraints::Equality> conval(n, m, goal);
  EXPECT_EQ(conval.OutputDimension(), goal->OutputDimension());

  // Some inputs
  Eigen::Vector4d x(4, 3, 2, 1);
  Eigen::Vector2d u(2, 3);

  // Evaluate method
  VectorXd c(n);
  VectorXd c2(n);
  Eigen::Vector4d c_expected(3, 1, -1, -3);
  goal->Evaluate(x, u, c);
  EXPECT_TRUE(c.isApprox(c_expected));
  conval.Evaluate(x, u, c2);
  EXPECT_TRUE(c2.isApprox(c_expected));

  // Jacobian method
  MatrixXd jac(n, n + m);
  MatrixXd jac2(n, n + m);
  MatrixXd jac_expected(n, n + m);
  jac_expected << MatrixXd::Identity(n, n), MatrixXd::Zero(n, m);
  goal->Jacobian(x, u, jac);
  EXPECT_TRUE(jac.isApprox(jac_expected));
  conval.Jacobian(x, u, jac2);
  EXPECT_TRUE(jac2.isApprox(jac_expected));
}

TEST(BasicConstraints, ControlBoundDimensions) {
  examples::ControlBound bnd({-1, -2}, {1, 2});
  EXPECT_EQ(bnd.StateDimension(), 0);
  EXPECT_EQ(bnd.ControlDimension(), 2);
}

TEST(BasicConstraints, GoalConstraintDimensions) {
  Eigen::Vector4d xf(1,2,3,4);
  examples::GoalConstraint goal(xf);
  EXPECT_EQ(goal.StateDimension(), 4);
  EXPECT_EQ(goal.ControlDimension(), 0);
}

TEST(StateBoundTest, EvaluateAndJacobian) {
    std::vector<double> lb = {-10.0, -5.0, -M_PI, -0.5, 0.0, -2.0};
    std::vector<double> ub = { 10.0,  5.0,  M_PI,  0.5, 20.0,  2.0};

    auto constraint = std::make_shared<altro::examples::StateBound>(lb, ub);

    EXPECT_EQ(constraint->StateDimension(), 6);
    EXPECT_EQ(constraint->ControlDimension(), 0);
    EXPECT_EQ(constraint->OutputDimension(), 12); // 6 lower + 6 upper

    Eigen::VectorXd x(6);
    x << 1, 2, 0.1, 0.2, 5, 0.5;
    Eigen::VectorXd u(2); // no control
    u << 0, 0;

    int32_t n_constraints = constraint->OutputDimension();
    Eigen::VectorXd c(12);
    constraint->Evaluate(x, u, c);

    // Check lower bounds: c[i] = lb[i] - x[i] <= 0
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(c(i), lb[i] - x(i), 1e-9);
    }
    // Check upper bounds: c[i+6] = x[i] - ub[i] <= 0
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(c(i + 6), x(i) - ub[i], 1e-9);
    }

    // Jacobian
    Eigen::MatrixXd jac_analytic(12, 6 + 2);
    constraint->Jacobian(x, u, jac_analytic);

    auto eval = [&](const Eigen::VectorXd& x_in, const Eigen::VectorXd& u_in, Eigen::Ref<Eigen::VectorXd> c_out) {
        constraint->Evaluate(x_in, u_in, c_out);
    };
    Eigen::MatrixXd jac_numeric = ComputeNumericalJacobian(eval, x, u, n_constraints);

    EXPECT_TRUE(jac_analytic.isApprox(jac_numeric, 1e-5));
}

TEST(LinearStateControlConstraintTest, SlantedRoadBoundaryInEgoFrame) {
    const int nx = 6;  // [px, py, theta, kappa, v, a]
    const int nu = 0;

    // Road normal vector: angle = 120 deg → (cos, sin) = (-0.5, 0.866)
    const double n_x = -0.5;
    const double n_y = std::sqrt(3.0) / 2.0;  // ≈ 0.8660254
    const double half_width = 1.5;  // lane half-width

    // Constraints:
    // Left:  n_x * x + n_y * y >= -half_width  →  -n_x * x - n_y * y <= half_width
    // Right: n_x * x + n_y * y <=  half_width

    Eigen::MatrixXd G(2, nx);
    G.setZero();
    // Row 0: left boundary → -n_x * x - n_y * y <= half_width
    G(0, 0) = -n_x;   // = 0.5
    G(0, 1) = -n_y;   // = -0.866...
    // Row 1: right boundary → n_x * x + n_y * y <= half_width
    G(1, 0) = n_x;    // = -0.5
    G(1, 1) = n_y;    // = 0.866...

    Eigen::MatrixXd H(2, nu);  // empty
    Eigen::VectorXd w(2);
    w << half_width, half_width;  // both bounds use same half-width

    auto constraint = std::make_shared<altro::examples::LinearStateControlConstraint>(G, H, w);

    EXPECT_EQ(constraint->StateDimension(), nx);
    EXPECT_EQ(constraint->ControlDimension(), nu);
    EXPECT_EQ(constraint->OutputDimension(), 2);

    // Test 1: vehicle at origin (0,0) → should be inside
    Eigen::VectorXd x(nx);
    x << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    Eigen::VectorXd u(nu);

    Eigen::VectorXd c(2);
    constraint->Evaluate(x, u, c);
    // c = G*x - w = [0 - 1.5, 0 - 1.5] = [-1.5, -1.5] → satisfied (<=0)
    EXPECT_NEAR(c(0), -1.5, 1e-9);
    EXPECT_NEAR(c(1), -1.5, 1e-9);

    // Test 2: vehicle on right edge: n·p = 1.5 → should be exactly on boundary
    // Solve: n_x * x + n_y * y = 1.5 → pick x=0 → y = 1.5 / n_y ≈ 1.732
    double y_on_right = half_width / n_y;  // ≈ 1.732
    x << 0.0, y_on_right, 0.0, 0.0, 0.0, 0.0;
    constraint->Evaluate(x, u, c);
    // Left: -n·p - 1.5 = -1.5 -1.5 = -3.0
    // Right: n·p - 1.5 = 1.5 - 1.5 = 0.0 → OK
    EXPECT_NEAR(c(0), -3.0, 1e-9);
    EXPECT_NEAR(c(1),  0.0, 1e-9);

    // Test 3: vehicle outside right: y = y_on_right + 0.1
    x(1) = y_on_right + 0.1;
    constraint->Evaluate(x, u, c);
    // n·p = n_y * (y_on_right + 0.1) = 1.5 + n_y*0.1 ≈ 1.5866
    // So c[1] = 1.5866 - 1.5 = 0.0866 > 0 → violated
    EXPECT_GT(c(1), 0.0);

    // Jacobian test
    Eigen::MatrixXd jac_analytic(2, nx + nu);
    constraint->Jacobian(x, u, jac_analytic);

    auto eval = [&](const Eigen::VectorXd& x_in, const Eigen::VectorXd& u_in,
                    Eigen::Ref<Eigen::VectorXd> c_out) {
        constraint->Evaluate(x_in, u_in, c_out);
    };
    Eigen::MatrixXd jac_numeric = ComputeNumericalJacobian(eval, x, u, 2, 1e-6);

    EXPECT_TRUE(jac_analytic.isApprox(jac_numeric, 1e-6));
}

TEST(CentripetalAccelerationConstraintTest, EvaluateAndJacobian) {
    const double a_max = 5.0;
    auto constraint = std::make_shared<altro::examples::CentripetalAccelerationConstraint>(a_max);

    // 检查维度
    EXPECT_EQ(constraint->StateDimension(), 6);
    EXPECT_EQ(constraint->ControlDimension(), 0);
    EXPECT_EQ(constraint->OutputDimension(), 2);

    // 构造状态: [px, py, theta, kappa, v, a]
    Eigen::VectorXd x(6);
    x << 0.0, 0.0, 0.0, 0.2, 3.0, 1.0;  // kappa=0.2, v=3 → a_c = 9 * 0.2 = 1.8
    Eigen::VectorXd u(0);  // no control

    // 手动计算预期值
    double v = x(4);
    double kappa = x(3);
    double a_c = v * v * kappa;  // = 9 * 0.2 = 1.8

    Eigen::VectorXd c_expected(2);
    c_expected(0) = a_c - a_max;   // 1.8 - 5.0 = -3.2
    c_expected(1) = -a_c - a_max;  // -1.8 - 5.0 = -6.8

    // 调用 Evaluate
    Eigen::VectorXd c(2);
    constraint->Evaluate(x, u, c);

    EXPECT_NEAR(c(0), c_expected(0), 1e-9);
    EXPECT_NEAR(c(1), c_expected(1), 1e-9);

    // 测试边界情况：a_c = a_max（应刚好满足约束）
    x(3) = a_max / (v * v);  // kappa = 5.0 / 9 ≈ 0.555...
    a_c = v * v * x(3);      // = 5.0
    constraint->Evaluate(x, u, c);
    EXPECT_NEAR(c(0), 0.0, 1e-9);        // 上界刚好接触
    EXPECT_NEAR(c(1), -2 * a_max, 1e-9); // -5 -5 = -10

    x(3) = 1.0;  // a_c = 9.0 > 5.0
    constraint->Evaluate(x, u, c);
    EXPECT_GT(c(0), 0.0);  // 违反上界
    EXPECT_LT(c(1), 0.0);  // 下界仍满足（因为 -9 -5 = -14 < 0）

    // Jacobian 测试
    auto eval = [&](const Eigen::VectorXd& x_in, const Eigen::VectorXd& u_in,
                    Eigen::Ref<Eigen::VectorXd> c_out) {
        constraint->Evaluate(x_in, u_in, c_out);
    };

    Eigen::MatrixXd jac_analytic(2, 6);  // 6 states, 0 controls → total 6 cols
    constraint->Jacobian(x, u, jac_analytic);

    Eigen::MatrixXd jac_numeric = ComputeNumericalJacobian(eval, x, u, 2, 1e-6);

    EXPECT_TRUE(jac_analytic.isApprox(jac_numeric, 1e-6));

    double dv2k_dv = 2.0 * v * x(3);     // ∂(v²κ)/∂v
    double dv2k_dkappa = v * v;          // ∂(v²κ)/∂κ

    EXPECT_NEAR(jac_analytic(0, 4), dv2k_dv, 1e-9);      // row 0, d/dv
    EXPECT_NEAR(jac_analytic(0, 3), dv2k_dkappa, 1e-9);  // row 0, d/dkappa
    EXPECT_NEAR(jac_analytic(1, 4), -dv2k_dv, 1e-9);     // row 1, d/dv
    EXPECT_NEAR(jac_analytic(1, 3), -dv2k_dkappa, 1e-9); // row 1, d/dkappa

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (j != 3 && j != 4) {
                EXPECT_NEAR(jac_analytic(i, j), 0.0, 1e-9);
            }
        }
    }
}

TEST(CentripetalJerkConstraintTest, BasicEvaluation) {
    double j_max = 10.0;
    auto constraint = std::make_shared<altro::examples::CentripetalJerkConstraint>(j_max);

    EXPECT_EQ(constraint->StateDimension(), 6);
    EXPECT_EQ(constraint->ControlDimension(), 2);
    EXPECT_EQ(constraint->OutputDimension(), 2);

    // 测试点 1: v=1, a=2, kappa=0.5, kappadot=1
    Eigen::VectorXd x(6);
    x << 0.0, 0.0, 0.0, 0.5, 1.0, 2.0;  // [px, py, theta, kappa, v, a]
    Eigen::VectorXd u(2);
    u << 1.0, 0.0;  // [kappadot, j]

    Eigen::VectorXd c(2);
    constraint->Evaluate(x, u, c);

    // 计算预期的 jc 值
    double v = x(4);
    double a = x(5);
    double kappa = x(3);
    double kappadot = u(0);
    double jc = 2.0 * v * a * kappa + v * v * kappadot;

    // 预期的约束值
    double expected_c0 = jc - j_max;
    double expected_c1 = -jc - j_max;

    EXPECT_NEAR(c(0), expected_c0, 1e-9);
    EXPECT_NEAR(c(1), expected_c1, 1e-9);

    // 测试点 2: 边界情况 v=0, a=0, kappa=0, kappadot=0
    x << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    u << 0.0, 0.0;

    constraint->Evaluate(x, u, c);

    expected_c0 = -j_max;
    expected_c1 = -j_max;

    EXPECT_NEAR(c(0), expected_c0, 1e-9);
    EXPECT_NEAR(c(1), expected_c1, 1e-9);

    // Jacobian 测试
    auto eval = [&](const Eigen::VectorXd& x_in, const Eigen::VectorXd& u_in, Eigen::Ref<Eigen::VectorXd> c_out) {
        constraint->Evaluate(x_in, u_in, c_out);
    };

    Eigen::MatrixXd jac_analytic(2, 8);  // 6 states + 2 controls
    constraint->Jacobian(x, u, jac_analytic);

    Eigen::MatrixXd jac_numeric = ComputeNumericalJacobian(eval, x, u, 2, 1e-6);

    EXPECT_TRUE(jac_analytic.isApprox(jac_numeric, 1e-6));
}

TEST(HeadingTrackingConstraintTest, ThirtyDegreeReferenceLine) {
  // --- 构造 30° 直线参考线 ---
  const int N = 100;
  const double total_length = 20.0;
  const double theta_ref_val = M_PI / 6.0;  // 30 degrees
  const double v_ref_val = 10.0;

  std::vector<Eigen::Vector4d> traj(N + 1);
  for (int k = 0; k <= N; ++k) {
    double s = total_length * static_cast<double>(k) / N;
    double x_ref = s * std::cos(theta_ref_val);
    double y_ref = s * std::sin(theta_ref_val);
    traj[k] = Eigen::Vector4d(x_ref, y_ref, theta_ref_val, v_ref_val);
  }

  auto ref_line = std::make_shared<altro::examples::ReferenceLine>(std::move(traj));
  auto projector = std::make_shared<altro::examples::ReferenceLineProjector>(ref_line);

  const double theta_max = M_PI / 12.0;  // 15 degrees tolerance
  auto constraint = std::make_shared<altro::examples::HeadingTrackingConstraint>(projector, theta_max);

  // 状态: [x, y, theta, kappa, v, a]
  Eigen::VectorXd x(6);
  Eigen::VectorXd u(0);  // unused

  // Case 1: Vehicle exactly aligned with reference (θ = 30°)
  x << 5.0, 2.5, theta_ref_val, 0.0, 8.0, 0.0;  // (5, 2.5) is on the line
  Eigen::VectorXd c(2);
  constraint->Evaluate(x, u, c);

  // diff = θ_vehicle - θ_ref = 0
  EXPECT_NEAR(c(0), -theta_max, 1e-9);  // 0 - θ_max
  EXPECT_NEAR(c(1), -theta_max, 1e-9);  // -0 - θ_max

  // Case 2: Vehicle heading = 40° → diff = 10° = π/18 ≈ 0.1745 rad < θ_max (15°=0.2618)
  x(2) = theta_ref_val + M_PI / 18.0;  // +10°
  constraint->Evaluate(x, u, c);
  double diff = M_PI / 18.0;
  EXPECT_NEAR(c(0), diff - theta_max, 1e-9);   // negative → satisfied
  EXPECT_NEAR(c(1), -diff - theta_max, 1e-9);  // negative

  // Case 3: Vehicle heading = 50° → diff = 20° > 15° → violate upper bound
  x(2) = theta_ref_val + M_PI / 9.0;  // +20°
  constraint->Evaluate(x, u, c);
  diff = M_PI / 9.0;  // ~0.349 > 0.2618
  EXPECT_GT(c(0), 0.0);   // violation!
  EXPECT_LT(c(1), 0.0);   // lower bound OK

  // Case 4: Vehicle far off the line but still projects to same theta_ref (since straight line)
  x << 100.0, -50.0, theta_ref_val - 0.2, 0.0, 5.0, 0.0;  // heading = 30° - 0.2 rad
  constraint->Evaluate(x, u, c);
  diff = -0.2;  // within ±0.2618
  EXPECT_NEAR(c(0), diff - theta_max, 1e-9);   // -0.2 - 0.2618 < 0
  EXPECT_NEAR(c(1), -diff - theta_max, 1e-9);  // +0.2 - 0.2618 ≈ -0.0618 < 0 → OK

  // Jacobian: only d/dtheta matters, and theta_ref is constant (straight line)
  Eigen::MatrixXd jac(2, 6);
  constraint->Jacobian(x, u, jac);

  Eigen::MatrixXd expected_jac = Eigen::MatrixXd::Zero(2, 6);
  expected_jac(0, 2) = 1.0;   // ∂(diff)/∂theta = 1
  expected_jac(1, 2) = -1.0;  // ∂(-diff)/∂theta = -1

  EXPECT_TRUE(jac.isApprox(expected_jac, 1e-9));
}

}  // namespace altro