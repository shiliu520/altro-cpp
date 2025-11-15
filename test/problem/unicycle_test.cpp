// Copyright [2021] Optimus Ride Inc.

#include <gtest/gtest.h>
#include <math.h>
#include <iostream>
#include <chrono>

#include "altro/problem/discretized_model.hpp"
#include "altro/utils/derivative_checker.hpp"
#include "altro/utils/benchmarking.hpp"
#include "examples/unicycle.hpp" 

namespace altro {
namespace examples {

TEST(UnicycleTest, Constructor) {
  Unicycle model;
  EXPECT_EQ(3, model.StateDimension());
  EXPECT_EQ(2, model.ControlDimension());
  EXPECT_TRUE(model.HasHessian());
}

TEST(UnicycleTest, Evaluate) {
  Unicycle model;
  double px = 1.0;
  double py = 2.0;
  double theta = M_PI / 3;
  double v = 0.1;
  double w = -0.3;
  VectorXd x = Eigen::Vector3d(px, py, theta);
  VectorXd u = Eigen::Vector2d(v, w);
  float t = 0.1;
  VectorXd xdot = model(x, u, t);
  VectorXd xdot_expected = Eigen::Vector3d(v * 0.5, v * sqrt(3) / 2.0, w);
  EXPECT_TRUE(xdot.isApprox(xdot_expected));
}

TEST(UnicycleTest, CheckJacobian) {
  Unicycle model;
  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(model.CheckJacobian());
  }
}

TEST(UnicycleTest, CheckHessian) {
  Unicycle model;
  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(model.CheckHessian(1e-4));
  }
}

TEST(UnicycleTest, BenchmarkRK4) {
  constexpr int NStates = Unicycle::NStates;
  constexpr int NControls = Unicycle::NControls;
  Unicycle model;
  problem::DiscretizedModel<Unicycle> dmodel(model);
  EXPECT_EQ(dmodel.GetIntegrator().StateMemorySize(), 3);
  EXPECT_EQ(dmodel.GetIntegrator().ControlMemorySize(), 2);

  VectorNd<NStates> x = VectorNd<NStates>::Random();
  VectorNd<NControls> u = VectorNd<NControls>::Random();
  VectorNd<NStates> xnext;
  const float t = 1.1;
  const float h = 0.1;

  auto eval = [&]() { dmodel.Evaluate(x, u, t, h, xnext); };
  fmt::print("\nIntegration\n");
  utils::Benchmark<std::chrono::microseconds>(eval, 2000).Print();

  fmt::print("\nJacobian\n");
  MatrixNxMd<NStates, NStates + NControls> jac;
  auto jacobian = [&]() { dmodel.Jacobian(x, u, t, h, jac); };
  utils::Benchmark<std::chrono::microseconds>(jacobian, 2000).Print();
}

TEST(UnicycleTest, RK4Fast_Fun) {
  constexpr int NStates = Unicycle::NStates;
  constexpr int NControls = Unicycle::NControls;
  Unicycle model;
  problem::DiscretizedModel<Unicycle, problem::RungeKutta4Fast<Unicycle::NStates, Unicycle::NControls>> dmodel(model);
  EXPECT_EQ(dmodel.GetIntegrator().StateMemorySize(), 3);
  EXPECT_EQ(dmodel.GetIntegrator().ControlMemorySize(), 2);

  VectorNd<NStates> x = VectorNd<NStates>::Random();
  VectorNd<NControls> u = VectorNd<NControls>::Random();
  VectorNd<NStates> xnext;
  const float t = 1.1;
  const float h = 0.1;

  auto eval = [&]() { dmodel.Evaluate(x, u, t, h, xnext); };
  fmt::print("\nIntegration\n");
  utils::Benchmark<std::chrono::microseconds>(eval, 2000).Print();

  fmt::print("\nJacobian\n");
  MatrixNxMd<NStates, NStates + NControls> jac;
  auto jacobian = [&]() { dmodel.Jacobian(x, u, t, h, jac); };
  utils::Benchmark<std::chrono::microseconds>(jacobian, 2000).Print();
}

template <typename DModel>
void ComputeFiniteDifferenceJacobian(
    DModel& dmodel, 
    const Eigen::VectorXd& x, 
    const Eigen::VectorXd& u, 
    float t, 
    float h, 
    Eigen::Ref<Eigen::MatrixXd> jac_fd, 
    double epsilon = 1e-6) // 扰动步长
{
    const int n = dmodel.StateDimension();
    const int m = dmodel.ControlDimension();
    const int N_total = n + m;
    
    // 联合状态/控制向量 z = [x, u]
    Eigen::VectorXd z(N_total);
    z.head(n) = x;
    z.tail(m) = u;
    
    // 工作空间
    Eigen::VectorXd z_plus = z;
    Eigen::VectorXd z_minus = z;
    Eigen::VectorXd xnext_plus(n);
    Eigen::VectorXd xnext_minus(n);
    
    // Jacobian 的列循环
    for (int i = 0; i < N_total; ++i) {
        
        // 1. 扰动
        z_plus(i) += epsilon;
        z_minus(i) -= epsilon;
        
        // 2. 计算积分：f(z + epsilon)
        // 注意：dmodel.Evaluate 接受 x, u
        dmodel.Evaluate(z_plus.head(n), z_plus.tail(m), t, h, xnext_plus);
        
        // 3. 计算积分：f(z - epsilon)
        dmodel.Evaluate(z_minus.head(n), z_minus.tail(m), t, h, xnext_minus);
        
        // 4. 计算中心差分逼近的列
        // J(:, i) = (f(z+) - f(z-)) / (2 * epsilon)
        jac_fd.col(i).noalias() = (xnext_plus - xnext_minus) / (2.0 * epsilon);
        
        // 5. 恢复扰动 (重要!)
        z_plus(i) = z(i);
        z_minus(i) = z(i);
    }
}

TEST(UnicycleTest, RK4Fast_Correctness) {
    constexpr int NStates = Unicycle::NStates;
    constexpr int NControls = Unicycle::NControls;
    Unicycle model;
    
    // NStates=3, NControls=2
    problem::DiscretizedModel<Unicycle, problem::RungeKutta4Fast<NStates, NControls>> dmodel(model);
    
    int n = dmodel.StateDimension(); // 3
    int m = dmodel.ControlDimension(); // 2
    
    // 随机初始化状态和控制
    VectorNd<NStates> x = VectorNd<NStates>::Random();
    VectorNd<NControls> u = VectorNd<NControls>::Random();
    
    const float t = 1.1;
    const float h = 0.1;
    
    // --- 1. Jacobian 正确性检验：与中心差分对比 ---
    
    MatrixNxMd<NStates, NStates + NControls> jac_rk4;
    dmodel.Jacobian(x, u, t, h, jac_rk4); // (1) 您的解析 RK4 Jacobian

    MatrixNxMd<NStates, NStates + NControls> jac_fd(n, n + m);
    // (2) 计算数值 Jacobian
    ComputeFiniteDifferenceJacobian(dmodel, x, u, t, h, jac_fd, 1e-7); 

    // 容差设置: 
    // RK4 Jacobian 对非线性项的逼近误差为 O(h^4)
    // FD 逼近的误差为 O(epsilon^2) + O(epsilon_machine)
    // 综合误差下，1e-6 或 1e-7 是合理的容差。
    EXPECT_TRUE(jac_rk4.isApprox(jac_fd, 1e-6)) 
        << "RK4 Jacobian failed Finite Difference Check:\n"
        << "FD:\n" << jac_fd << "\n"
        << "Analytic (RK4):\n" << jac_rk4;
    
    // --- 2. 积分结果的简单烟雾测试 (可选但推荐) ---
    VectorNd<NStates> xnext_rk4;
    dmodel.Evaluate(x, u, t, h, xnext_rk4);

    // 确保积分结果与欧拉法在小h下接近，作为基本正确性检查
    Eigen::VectorXd xdot_cont(n);
    model.Evaluate(x, u, t, xdot_cont);
    Eigen::VectorXd xnext_euler = x + xdot_cont * h;
    
    // RK4 应该比欧拉更准确，但我们仅检查它们是否足够接近 (小h下)
    // EXPECT_FALSE(xnext_rk4.isApprox(xnext_euler, 1e-4)); // 确保它不等于欧拉结果
    // EXPECT_TRUE(xnext_rk4.isApprox(xnext_euler, 1e-1));  // 确保它在合理的范围内
}

}  // namespace examples
}  // namespace altro