// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <vector>
#include <array>
#include <memory>

#include "altro/eigentypes.hpp"
#include "altro/problem/dynamics.hpp"
#include "altro/common/state_control_sized.hpp"
namespace altro {
namespace problem {
/**
 * @brief Interface class for explicit integration methods for dynamical
 * systems.
 *
 * All sub-classes must implement the `Integrate` method that integrates an
 * arbitrary functor over some time step, as well as it's first derivative
 * via the `Jacobian` method. 
 * 
 * Sub-classes should have a constructor that takes the state and control 
 * dimension, e.g.:
 * 
 * `MyIntegrator(int n, int m);`
 *
 * @tparam DynamicsFunc the type of the function-like object that evaluates a
 * first-order ordinary differential equation with the following signature:
 * dynamics(const VectorXd& x, const VectorXd& u, float t) const
 *
 * See `ContinuousDynamics` class for the expected interface.
 */
template <int NStates, int NControls>
class ExplicitIntegrator : public StateControlSized<NStates, NControls> {
 protected:
  using DynamicsPtr = std::shared_ptr<ContinuousDynamics>;

 public:
  ExplicitIntegrator(int n, int m) : StateControlSized<NStates, NControls>(n, m) {}
  ExplicitIntegrator() : StateControlSized<NStates, NControls>() {

  }
  virtual ~ExplicitIntegrator() = default;

  /**
   * @brief Integrate the dynamics over a given time step
   *
   * @param[in] dynamics ContinuousDynamics object to evaluate the continuous
   * dynamics
   * @param[in] x state vector
   * @param[in] u control vector
   * @param[in] t independent variable (e.g. time)
   * @param[in] h discretization step length (e.g. time step)
   * @return VectorXd state vector at the end of the time step
   */
  virtual void Integrate(const DynamicsPtr& dynamics, const VectorXdRef& x,
                         const VectorXdRef& u, float t, float h,
                         Eigen::Ref<VectorXd> xnext) = 0;

  /**
   * @brief Evaluate the Jacobian of the discrete dynamics
   *
   * Will typically call the continuous dynamics Jacobian.
   *
   * @pre Jacobian must be initialized
   *
   * @param[in] dynamics ContinuousDynamics object to evaluate the continuous
   * dynamics
   * @param[in] x state vector
   * @param[in] u control vector
   * @param[in] t independent variable (e.g. time)
   * @param[in] h discretization step length (e.g. time step)
   * @param[out] jac discrete dynamics Jacobian evaluated at x, u, t.
   */
  virtual void Jacobian(const DynamicsPtr& dynamics, const VectorXdRef& x,
                        const VectorXdRef& u, float t, float h, Eigen::Ref<MatrixXd> jac) = 0;
};

/**
 * @brief Basic explicit Euler integration
 *
 * Simplest integrator that requires only a single evaluationg of the continuous
 * dynamics but suffers from significant integration errors.
 *
 * @tparam DynamicsFunc
 */
class ExplicitEuler final : public ExplicitIntegrator<Eigen::Dynamic, Eigen::Dynamic> {
 public:
  ExplicitEuler(int n, int m) : ExplicitIntegrator<Eigen::Dynamic, Eigen::Dynamic>(n, m) {}
  void Integrate(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                 float t, float h, Eigen::Ref<VectorXd> xnext) override {
    dynamics->Evaluate(x, u, t,  xnext);
    xnext = x + xnext * h;
  }
  void Jacobian(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                float t, float h, Eigen::Ref<MatrixXd> jac) override {
    int n = x.size();
    int m = u.size();
    dynamics->Jacobian(x, u, t, jac);
    jac = MatrixXd::Identity(n, n + m) + jac * h;
  }
};

/**
 * @brief Fourth-order explicit Runge Kutta integrator.
 *
 * De-facto explicit integrator for many robotics applications.
 * Good balance between accuracy and computational effort.
 *
 * @tparam DynamicsFunc
 */
template <int NStates, int NControls>
class RungeKutta4 final : public ExplicitIntegrator<NStates, NControls> {
  using typename ExplicitIntegrator<NStates, NControls>::DynamicsPtr;
 public:

  RungeKutta4(int n, int m) : ExplicitIntegrator<NStates, NControls>(n, m) {
    Init();
  }
  RungeKutta4() : ExplicitIntegrator<NStates, NControls>() {
    Init();
  }
  void Integrate(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                 float t, float h, Eigen::Ref<VectorXd> xnext) override {

    dynamics->Evaluate(x, u, t, k1_);
    dynamics->Evaluate(x + k1_ * 0.5 * h, u, t + 0.5 * h, k2_);  // NOLINT(readability-magic-numbers)
    dynamics->Evaluate(x + k2_ * 0.5 * h, u, t + 0.5 * h, k3_);  // NOLINT(readability-magic-numbers)
    dynamics->Evaluate(x + k3_ * h, u, t + h, k4_);
    xnext = x + h * (k1_ + 2 * k2_ + 2 * k3_ + k4_) / 6;  // NOLINT(readability-magic-numbers)
  }
  void Jacobian(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                float t, float h, Eigen::Ref<MatrixXd> jac) override {
    int n = dynamics->StateDimension();
    int m = dynamics->ControlDimension();

    dynamics->Evaluate(x, u, t, k1_);
    dynamics->Evaluate(x + k1_ * 0.5 * h, u, t + 0.5 * h, k2_);  // NOLINT(readability-magic-numbers)
    dynamics->Evaluate(x + k2_ * 0.5 * h, u, t + 0.5 * h, k3_);  // NOLINT(readability-magic-numbers)

    dynamics->Jacobian(x, u, t, jac);
    A_[0] = jac.topLeftCorner(n, n);
    B_[0] = jac.topRightCorner(n, m);
    dynamics->Jacobian(x + 0.5 * k1_ * h, u, 0.5 * t, jac);  // NOLINT(readability-magic-numbers)
    A_[1] = jac.topLeftCorner(n, n);
    B_[1] = jac.topRightCorner(n, m);
    dynamics->Jacobian(x + 0.5 * k2_ * h, u, 0.5 * t, jac);  // NOLINT(readability-magic-numbers)
    A_[2] = jac.topLeftCorner(n, n);
    B_[2] = jac.topRightCorner(n, m);
    dynamics->Jacobian(x + k3_ * h, u, t, jac);
    A_[3] = jac.topLeftCorner(n, n);
    B_[3] = jac.topRightCorner(n, m);

    dA_[0] = A_[0] * h;
    dA_[1] = A_[1] * (MatrixXd::Identity(n, n) + 0.5 * dA_[0]) * h;  // NOLINT(readability-magic-numbers)
    dA_[2] = A_[2] * (MatrixXd::Identity(n, n) + 0.5 * dA_[1]) * h;  // NOLINT(readability-magic-numbers)
    dA_[3] = A_[3] * (MatrixXd::Identity(n, n) + dA_[2]) * h;

    dB_[0] = B_[0] * h;
    dB_[1] = B_[1] * h + 0.5 * A_[1] * dB_[0] * h;  // NOLINT(readability-magic-numbers)
    dB_[2] = B_[2] * h + 0.5 * A_[2] * dB_[1] * h;  // NOLINT(readability-magic-numbers)
    dB_[3] = B_[3] * h + A_[3] * dB_[2] * h;

    jac.topLeftCorner(n, n) =
        MatrixXd::Identity(n, n)
        + (dA_[0] + 2 * dA_[1] + 2 * dA_[2] + dA_[3]) / 6;  // NOLINT(readability-magic-numbers)
    jac.topRightCorner(n, m) =
        (dB_[0] + 2 * dB_[1] + 2 * dB_[2] + dB_[3]) / 6;  // NOLINT(readability-magic-numbers)
  }

 private:
  void Init() {
    int n = this->StateDimension();
    int m = this->ControlDimension();
    k1_.setZero(n);
    k2_.setZero(n);
    k3_.setZero(n);
    k4_.setZero(n);
    for (int i = 0; i < 4; ++i) {
      A_[i].setZero(n, n); 
      B_[i].setZero(n, m);
      dA_[i].setZero(n, n); 
      dB_[i].setZero(n, m);
    }
  }

  // These need to be mutable to keep the integration methods as const methods
  // Since they replace arrays that would otherwise be created temporarily and 
  // provide no public access, it should be fine.
  VectorNd<NStates> k1_;
  VectorNd<NStates> k2_;
  VectorNd<NStates> k3_;
  VectorNd<NStates> k4_;
  std::array<MatrixNxMd<NStates, NStates>, 4> A_;
  std::array<MatrixNxMd<NStates, NControls>, 4> B_;
  std::array<MatrixNxMd<NStates, NStates>, 4> dA_;
  std::array<MatrixNxMd<NStates, NControls>, 4> dB_;
};

/**
 * @brief Fourth-order explicit Runge Kutta integrator with optimized memory usage.
 * 
 * De-facto explicit integrator for many robotics applications.
 * Good balance between accuracy and computational effort.
 * This implementation minimizes temporary object creation for better performance.
 * 
 * @tparam NStates State dimension
 * @tparam NControls Control dimension
 */
template <int NStates, int NControls>
class RungeKutta4Fast final : public ExplicitIntegrator<NStates, NControls> {
  using typename ExplicitIntegrator<NStates, NControls>::DynamicsPtr;
  
public:
  // RK4 coefficients as compile-time constants
  static constexpr double STAGE_WEIGHT_2 = 0.5;
  static constexpr double STAGE_WEIGHT_3 = 0.5; 
  static constexpr double STAGE_WEIGHT_4 = 1.0;
  static constexpr double RESULT_WEIGHT_1 = 1.0 / 6.0;
  static constexpr double RESULT_WEIGHT_2 = 1.0 / 3.0;
  static constexpr double RESULT_WEIGHT_3 = 1.0 / 3.0;
  static constexpr double RESULT_WEIGHT_4 = 1.0 / 6.0;

  RungeKutta4Fast(int n, int m) : ExplicitIntegrator<NStates, NControls>(n, m) {
    Init();
  }
  
  RungeKutta4Fast() : ExplicitIntegrator<NStates, NControls>() {
    Init();
  }

  /**
   * @brief Integrate the dynamics using RK4 method
   * 
   * @param dynamics System dynamics function
   * @param x Current state
   * @param u Control input  
   * @param t Current time
   * @param h Time step
   * @param xnext Next state (output)
   */
  void Integrate(const DynamicsPtr& dynamics, const VectorXdRef& x, 
                 const VectorXdRef& u, float t, float h, 
                 Eigen::Ref<VectorXd> xnext) override {
    
    // Validate inputs
    assert(x.size() == this->StateDimension());
    assert(u.size() == this->ControlDimension());
    assert(xnext.size() == this->StateDimension());
    
    const double half_h = STAGE_WEIGHT_2 * h;  // 0.5 * h
    
    // Stage 1: k1 = f(t, x)
    dynamics->Evaluate(x, u, t, k1_);
    
    // Stage 2: k2 = f(t + h/2, x + h/2 * k1)
    temp_state_.noalias() = k1_ * half_h;
    temp_state_ += x;
    dynamics->Evaluate(temp_state_, u, t + half_h, k2_);
    
    // Stage 3: k3 = f(t + h/2, x + h/2 * k2)  
    temp_state_.noalias() = k2_ * half_h;
    temp_state_ = x + temp_state_;  // Use expression template
    dynamics->Evaluate(temp_state_, u, t + half_h, k3_);
    
    // Stage 4: k4 = f(t + h, x + h * k3)
    temp_state_.noalias() = k3_ * h;
    temp_state_ = x + temp_state_;
    dynamics->Evaluate(temp_state_, u, t + h, k4_);
    
    // Combine results: xnext = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    xnext.noalias() = k1_ * RESULT_WEIGHT_1;
    xnext.noalias() += k2_ * RESULT_WEIGHT_2;
    xnext.noalias() += k3_ * RESULT_WEIGHT_3; 
    xnext.noalias() += k4_ * RESULT_WEIGHT_4;
    xnext *= h;
    xnext += x;
  }

  /**
   * @brief Compute the Jacobian of the RK4 integration step
   * 
   * @param dynamics System dynamics function
   * @param x Current state
   * @param u Control input
   * @param t Current time  
   * @param h Time step
   * @param jac Jacobian matrix (output)
   */
void Jacobian(const DynamicsPtr& dynamics, const VectorXdRef& x,
              const VectorXdRef& u, float t, float h,
              Eigen::Ref<MatrixXd> jac) override {
    
    const int n = dynamics->StateDimension();
    const int m = dynamics->ControlDimension();
    const double half_h = STAGE_WEIGHT_2 * h;
    
    // Validate dimensions
    assert(x.size() == n);
    assert(u.size() == m);
    assert(jac.rows() == n);
    assert(jac.cols() == n + m);
    
    // Compute intermediate states for Jacobian evaluation
    dynamics->Evaluate(x, u, t, k1_);
    
    temp_state_.noalias() = k1_ * half_h;
    temp_state_ += x;
    dynamics->Evaluate(temp_state_, u, t + half_h, k2_);
    
    temp_state_.noalias() = k2_ * half_h;
    temp_state_ = x + temp_state_;
    dynamics->Evaluate(temp_state_, u, t + half_h, k3_);
    
    // Compute incremental Jacobians using noalias()
    // Stage 1: f(x, u, t)
    dynamics->Jacobian(x, u, t, jac);
    A_[0] = jac.topLeftCorner(n, n);
    B_[0] = jac.topRightCorner(n, m);
    
    // Stage 2: f(x + h/2*k1, u, t + h/2)
    temp_state_.noalias() = x + k1_ * half_h; // 使用 k1_ 的结果，与积分步保持一致
    dynamics->Jacobian(temp_state_, u, t + half_h, jac);
    A_[1] = jac.topLeftCorner(n, n);
    B_[1] = jac.topRightCorner(n, m);
    
    // Stage 3: f(x + h/2*k2, u, t + h/2)
    temp_state_.noalias() = x + k2_ * half_h; // 使用 k2_ 的结果
    dynamics->Jacobian(temp_state_, u, t + half_h, jac);
    A_[2] = jac.topLeftCorner(n, n);
    B_[2] = jac.topRightCorner(n, m);
    
    // Stage 4: f(x + h*k3, u, t + h)
    temp_state_.noalias() = x + k3_ * h; // 使用 k3_ 的结果
    dynamics->Jacobian(temp_state_, u, t + h, jac);
    A_[3] = jac.topLeftCorner(n, n);
    B_[3] = jac.topRightCorner(n, m);
    
    // --- 3. Compute incremental Jacobians (dKi/dx and dKi/du) using CORRECT recursion ---
    const MatrixXd I = MatrixXd::Identity(n, n);
    const double HALF_H = STAGE_WEIGHT_2 * h; // STAGE_WEIGHT_2 * h
    const double FULL_H = STAGE_WEIGHT_4 * h; // STAGE_WEIGHT_4 * h

    // K1 Jacobians (dA_[0] = dK1/dx, dB_[0] = dK1/du)
    // dK1/dx = A0; dK1/du = B0
    dA_[0].noalias() = A_[0];
    dB_[0].noalias() = B_[0];

    // K2 Jacobians (dK2/dx, dK2/du)
    // dK2/dx = A1 * (I + h/2 * dK1/dx)
    dA_[1].noalias() = A_[1] * (I + HALF_H * dA_[0]);
    // dK2/du = B1 + A1 * (h/2 * dK1/du)
    dB_[1].noalias() = B_[1] + A_[1] * (HALF_H * dB_[0]);

    // K3 Jacobians (dK3/dx, dK3/du)
    // dK3/dx = A2 * (I + h/2 * dK2/dx)
    dA_[2].noalias() = A_[2] * (I + HALF_H * dA_[1]);
    // dK3/u = B2 + A2 * (h/2 * dK2/du)
    dB_[2].noalias() = B_[2] + A_[2] * (HALF_H * dB_[1]);

    // K4 Jacobians (dK4/dx, dK4/du)
    // dK4/dx = A3 * (I + h * dK3/dx)
    dA_[3].noalias() = A_[3] * (I + FULL_H * dA_[2]);
    // dK4/u = B3 + A3 * (h * dK3/du)
    dB_[3].noalias() = B_[3] + A_[3] * (FULL_H * dB_[2]);

    // --- 4. Combine results: Phi = d(xnext)/dx and Psi = d(xnext)/du ---
    // Phi = I + h/6 * (dK1/dx + 2*dK2/dx + 2*dK3/dx + dK4/dx)
    jac.topLeftCorner(n, n).noalias() =
        I + h * (RESULT_WEIGHT_1 * dA_[0] +
                 RESULT_WEIGHT_2 * dA_[1] +
                 RESULT_WEIGHT_3 * dA_[2] +
                 RESULT_WEIGHT_4 * dA_[3]);

    // Psi = h/6 * (dK1/du + 2*dK2/du + 2*dK3/du + dK4/du)
    jac.topRightCorner(n, m).noalias() =
        h * (RESULT_WEIGHT_1 * dB_[0] +
             RESULT_WEIGHT_2 * dB_[1] +
             RESULT_WEIGHT_3 * dB_[2] +
             RESULT_WEIGHT_4 * dB_[3]);
}

  // Delete copy operations to prevent accidental sharing of workspace
  RungeKutta4Fast(const RungeKutta4Fast&) = delete;
  RungeKutta4Fast& operator=(const RungeKutta4Fast&) = delete;
  
  // Allow move operations if needed
  RungeKutta4Fast(RungeKutta4Fast&&) = default;
  RungeKutta4Fast& operator=(RungeKutta4Fast&&) = default;

private:
  void Init() {
    const int n = this->StateDimension();
    const int m = this->ControlDimension();
    
    k1_.setZero(n);
    k2_.setZero(n);
    k3_.setZero(n);
    k4_.setZero(n);
    temp_state_.setZero(n);
    
    for (int i = 0; i < 4; ++i) {
      A_[i].setZero(n, n);
      B_[i].setZero(n, m);
      dA_[i].setZero(n, n);
      dB_[i].setZero(n, m);
    }
  }

  // Workspace variables - mutable to maintain const correctness in interface
  VectorNd<NStates> k1_;
  VectorNd<NStates> k2_; 
  VectorNd<NStates> k3_;
  VectorNd<NStates> k4_;
  VectorNd<NStates> temp_state_;  // Reusable temporary for intermediate states
  
  std::array<MatrixNxMd<NStates, NStates>, 4> A_;
  std::array<MatrixNxMd<NStates, NControls>, 4> B_;
  std::array<MatrixNxMd<NStates, NStates>, 4> dA_;
  std::array<MatrixNxMd<NStates, NControls>, 4> dB_;
};

template <int NStates, int NControls>
constexpr double RungeKutta4Fast<NStates, NControls>::STAGE_WEIGHT_2;
template <int NStates, int NControls>
constexpr double RungeKutta4Fast<NStates, NControls>::STAGE_WEIGHT_3;
template <int NStates, int NControls>
constexpr double RungeKutta4Fast<NStates, NControls>::STAGE_WEIGHT_4;
template <int NStates, int NControls>
constexpr double RungeKutta4Fast<NStates, NControls>::RESULT_WEIGHT_1;
template <int NStates, int NControls>
constexpr double RungeKutta4Fast<NStates, NControls>::RESULT_WEIGHT_2;
template <int NStates, int NControls>
constexpr double RungeKutta4Fast<NStates, NControls>::RESULT_WEIGHT_3;
template <int NStates, int NControls>
constexpr double RungeKutta4Fast<NStates, NControls>::RESULT_WEIGHT_4;

}  // namespace problem
}  // namespace altro