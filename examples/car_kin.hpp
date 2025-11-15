#pragma once

#include "altro/eigentypes.hpp"
#include "altro/problem/dynamics.hpp"

namespace altro {
namespace examples {

/**
 * @brief Simple kinematic car model with Ackermann steering and longitudinal acceleration.
 *
 * States:
 *   x(0): X position [m]
 *   x(1): Y position [m]
 *   x(2): Yaw angle [rad]
 *   x(3): Longitudinal velocity [m/s]
 *
 * Controls:
 *   u(0): Steering angle (front wheel) [rad]
 *   u(1): Longitudinal acceleration [m/s^2]
 *
 * Dynamics:
 *   xdot(0) = v * cos(yaw)
 *   xdot(1) = v * sin(yaw)
 *   xdot(2) = v / L * tan(steer)
 *   xdot(3) = accel
 */
class CarKinematic : public problem::ContinuousDynamics {
 public:
  CarKinematic(double length = 4.368,   // [m]
               double width = 1.823,    // [m]
               double height = 1.483,   // [m]
               double wheelbase = 2.648,// [m]
               double rear_overhang = 0.800)  // [m]
      : length_(length),
        width_(width),
        height_(height),
        wheelbase_(wheelbase),
        rear_overhang_(rear_overhang) {
    front_overhang_ = length_ - (rear_overhang_ + wheelbase_);
  }

  static constexpr int NStates = 4;
  static constexpr int NControls = 2;

  int StateDimension() const override { return NStates; }
  int ControlDimension() const override { return NControls; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, const float t,
                       Eigen::Ref<VectorXd> xdot) override;
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, const float t,
                Eigen::Ref<MatrixXd> jac) override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const float t,
               const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) override;

  bool HasHessian() const override { return true; }

  // Vehicle parameters (public read-only)
  double length() const { return length_; }
  double width() const { return width_; }
  double height() const { return height_; }
  double wheelbase() const { return wheelbase_; }
  double rear_overhang() const { return rear_overhang_; }
  double front_overhang() const { return front_overhang_; }

 private:
  double length_;
  double width_;
  double height_;
  double wheelbase_;
  double rear_overhang_;
  double front_overhang_;
};

/**
 * @brief Extended kinematic car model with longitudinal jerk and curvature rate control.
 *
 * States:
 *   x(0): X position [m]
 *   x(1): Y position [m]
 *   x(2): Yaw angle [rad]
 *   x(3): Curvature κ [1/m]
 *   x(4): Longitudinal velocity v [m/s]
 *   x(5): Longitudinal acceleration a [m/s^2]
 *
 * Controls:
 *   u(0): Curvature rate (dot κ) [1/m/s]
 *   u(1): Longitudinal jerk j [m/s^3]
 *
 * Dynamics:
 *   xdot(0) = v * cos(yaw)
 *   xdot(1) = v * sin(yaw)
 *   xdot(2) = v * κ
 *   xdot(3) = u0
 *   xdot(4) = a
 *   xdot(5) = u1
 */
class CarExtended : public problem::ContinuousDynamics {
 public:
  CarExtended(double length = 4.368,
              double width = 1.823,
              double height = 1.483,
              double wheelbase = 2.648,
              double rear_overhang = 0.800);

  static constexpr int NStates = 6;
  static constexpr int NControls = 2;

  int StateDimension() const override { return NStates; }
  int ControlDimension() const override { return NControls; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, const float t,
                Eigen::Ref<VectorXd> xdot) override;
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, const float t,
                Eigen::Ref<MatrixXd> jac) override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const float t,
               const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) override;
  bool HasHessian() const override { return true; }

  // Same parameter accessors as CarKinematic
  double length() const { return length_; }
  double width() const { return width_; }
  // ... etc ...

 private:
  double length_, width_, height_, wheelbase_, rear_overhang_, front_overhang_;
};

}  // namespace examples
}  // namespace altro