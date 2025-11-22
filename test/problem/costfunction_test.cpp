// Copyright [2021] Optimus Ride Inc.

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>

#include "altro/common/functionbase.hpp"
#include "altro/problem/costfunction.hpp"
#include "examples/reference_line.hpp"
namespace altro {
// Needed to prevent linker errors
constexpr int ScalarFunction::NOutputs;

namespace problem {

class TestCostFunction : public CostFunction {
 public:
  // Provide access to optional ScalarFunction API
  using ScalarFunction::Gradient;
  using ScalarFunction::Hessian;

  static constexpr int NStates = 4;
  static constexpr int NControls = 2;
  int StateDimension() const override { return NStates; }
  int ControlDimension() const override { return NControls; }
  
  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override {
    return x.squaredNorm() + u.squaredNorm();
  }
  void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                Eigen::Ref<VectorXd> du) override {
    dx = 2 * x;
    du = 2 * u;
  }
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
                       Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu) override {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    dxdx.setIdentity();
    dudu.setIdentity();
    dxdu.setZero();
  }
};

// Needed to prevent linker errors
constexpr int TestCostFunction::NStates;
constexpr int TestCostFunction::NControls;
const double kTol = 1e-6;

// Helper: create ReferenceLine from vector
std::shared_ptr<altro::examples::ReferenceLine> MakeRefLine(const std::vector<Eigen::Vector4d>& traj) {
    return std::make_shared<altro::examples::ReferenceLine>(traj);
}


TEST(FunctionBase, CostFunSizes) {
  TestCostFunction costfun;
  EXPECT_EQ(costfun.NStates, 4);
  EXPECT_EQ(costfun.NControls, 2);
  EXPECT_EQ(costfun.NOutputs, 1);
  EXPECT_EQ(costfun.StateDimension(), 4);
  EXPECT_EQ(costfun.ControlDimension(), 2);
  EXPECT_EQ(costfun.OutputDimension(), 1);
}

TEST(FunctionBase, CostFunEval) {
  TestCostFunction costfun;
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  VectorXd u = Eigen::Vector2d(5, 6);
  double J = costfun.Evaluate(x,u);
  const double J_expected = 1 + 4 + 9 + 16 + 25 + 36;
  EXPECT_DOUBLE_EQ(J, J_expected);
}

TEST(FunctionBase, CostFunGradient) {
  TestCostFunction costfun;
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  VectorXd u = Eigen::Vector2d(5, 6);

  VectorXd dx(costfun.StateDimension());
  VectorXd du(costfun.ControlDimension());
  costfun.Gradient(x, u, dx, du);
  EXPECT_TRUE(dx.isApprox(2 * x));
  EXPECT_TRUE(du.isApprox(2 * u));

  VectorXd grad = VectorXd::Zero(dx.size() + du.size());
  VectorXd grad_expected = grad;
  grad_expected << dx, du;
  costfun.Gradient(x, u, grad);
  EXPECT_TRUE(grad.isApprox(grad_expected));

  for (int i = 0; i < 10; ++i) {
    costfun.CheckGradient();
  }
}

TEST(FunctionBase, CostFunHessian) {
  TestCostFunction costfun;
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  VectorXd u = Eigen::Vector2d(5, 6);

  const int n = costfun.StateDimension();
  const int m = costfun.ControlDimension();
  MatrixXd dxdx(n, n);
  MatrixXd dxdu(n, m);
  MatrixXd dudu(m, m);
  costfun.Hessian(x, u, dxdx, dxdu, dudu);
  EXPECT_TRUE(dxdx.isApprox(MatrixXd::Identity(n, n)));
  EXPECT_TRUE(dudu.isApprox(MatrixXd::Identity(m, m)));
  EXPECT_TRUE(dxdu.isApproxToConstant(0));

  MatrixXd hess = MatrixXd::Zero(n + m, n + m);
  MatrixXd hess_expected = hess;
  hess_expected << dxdx, dxdu, dxdu.transpose(), dudu;
  costfun.Hessian(x, u, hess);
  EXPECT_TRUE(hess.isApprox(hess_expected));

  for (int i = 0; i < 10; ++i) {
    costfun.CheckHessian();
  }
}

TEST(ReferenceLineTest, SinglePointTrajectory) {
    std::vector<Eigen::Vector4d> traj(1);
    traj[0] << 1.0, 2.0, M_PI / 3.0, 5.0;
    auto ref_line = MakeRefLine(traj);

    Eigen::Vector2d vehicle_pos(-1.0, -1.0);
    auto res = ref_line->Project(vehicle_pos, 0);

    EXPECT_NEAR(res.pos.x(), 1.0, kTol);
    EXPECT_NEAR(res.pos.y(), 2.0, kTol);
    EXPECT_NEAR(res.theta, M_PI / 3.0, kTol);
    EXPECT_NEAR(res.vel, 5.0, kTol);
    EXPECT_EQ(res.next_index_hint, 0);
}

TEST(ReferenceLineTest, TwoPointsStraight_LineStart) {
    std::vector<Eigen::Vector4d> traj(2);
    traj[0] << 0.0, 0.0, M_PI / 4.0, 1.0;
    traj[1] << 2.0, 2.0, M_PI / 4.0, 2.0;
    auto ref_line = MakeRefLine(traj);

    Eigen::Vector2d vehicle_pos(-1.0, -1.0);
    auto res = ref_line->Project(vehicle_pos, 0);

    EXPECT_NEAR(res.pos.x(), 0.0, kTol);
    EXPECT_NEAR(res.pos.y(), 0.0, kTol);
    EXPECT_NEAR(res.theta, M_PI / 4.0, kTol);
    EXPECT_NEAR(res.vel, 1.0, kTol);
    EXPECT_EQ(res.next_index_hint, 0);
}

TEST(ReferenceLineTest, TwoPointsStraight_Middle) {
    std::vector<Eigen::Vector4d> traj(2);
    traj[0] << 0.0, 0.0, M_PI / 4.0, 1.0;
    traj[1] << 2.0, 2.0, M_PI / 4.0, 2.0;
    auto ref_line = MakeRefLine(traj);

    Eigen::Vector2d vehicle_pos(0.25, 0.3);
    auto res = ref_line->Project(vehicle_pos, 0);

    // Compute expected projection onto segment (0,0)->(2,2)
    Eigen::Vector2d A(0.0, 0.0), B(2.0, 2.0);
    Eigen::Vector2d AB = B - A;
    double t = (vehicle_pos - A).dot(AB) / AB.squaredNorm();
    t = (t < 0.0) ? 0.0 : (t > 1.0) ? 1.0 : t;
    Eigen::Vector2d expected_pos = A + t * AB;
    double expected_theta = M_PI / 4.0;
    double expected_vel = 1.0 + t * (2.0 - 1.0);

    EXPECT_NEAR(res.pos.x(), expected_pos.x(), kTol);
    EXPECT_NEAR(res.pos.y(), expected_pos.y(), kTol);
    EXPECT_NEAR(res.theta, expected_theta, kTol);
    EXPECT_NEAR(res.vel, expected_vel, kTol);
    EXPECT_EQ(res.next_index_hint, 0);
}

TEST(ReferenceLineTest, TwoPointsStraight_End) {
    std::vector<Eigen::Vector4d> traj(2);
    traj[0] << 0.0, 0.0, M_PI / 4.0, 1.0;
    traj[1] << 2.0, 2.0, M_PI / 4.0, 2.0;
    auto ref_line = MakeRefLine(traj);

    Eigen::Vector2d vehicle_pos(2.5, 3.0);
    auto res = ref_line->Project(vehicle_pos, 0);

    EXPECT_NEAR(res.pos.x(), 2.0, kTol);
    EXPECT_NEAR(res.pos.y(), 2.0, kTol);
    EXPECT_NEAR(res.theta, M_PI / 4.0, kTol);
    EXPECT_NEAR(res.vel, 2.0, kTol);
    EXPECT_EQ(res.next_index_hint, 0);
}

TEST(ReferenceLineTest, UTurnAngleWrapping) {
    std::vector<Eigen::Vector4d> traj(2);
    // -1 degree and +1 degree in radians
    double deg2rad = M_PI / 180.0;
    double theta1 = -1.0 * deg2rad;   // ≈ -0.01745 rad
    double theta2 =  1.0 * deg2rad;   // ≈  0.01745 rad

    traj[0] << -0.1, -0.99, theta1, 0.0;
    traj[1] <<  0.1, -0.99, theta2, 0.0;
    auto ref_line = MakeRefLine(traj);

    Eigen::Vector2d vehicle_pos(0.0, 0.0); // above the segment
    auto res = ref_line->Project(vehicle_pos, 0);

    // Expected projection: (0.0, -0.99) → t = 0.5
    EXPECT_NEAR(res.pos.x(), 0.0, kTol);
    EXPECT_NEAR(res.pos.y(), -0.99, kTol);

    // Interpolated angle should be ~0 rad (not π!)
    // If your Project uses linear interpolation without wrap, this will FAIL
    // → which is good! It tells you to implement angle-aware interpolation.
    EXPECT_NEAR(res.theta, 0.0, 1e-3); // 0.001 rad ≈ 0.057 degrees

    EXPECT_NEAR(res.vel, 0.0, kTol);
    EXPECT_EQ(res.next_index_hint, 0);
}

}  // namespace problem
}  // namespace altro