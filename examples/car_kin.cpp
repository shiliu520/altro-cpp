#include <cmath>

#include "altro/utils/utils.hpp"
#include "examples/car_kin.hpp"

namespace altro {
namespace examples {

void CarKinematic::Evaluate(const VectorXdRef& x, const VectorXdRef& u, const float t,
                            Eigen::Ref<VectorXd> xdot) {
  ALTRO_UNUSED(t);
  double yaw = x(2);
  double v = x(3);
  double steer = u(0);
  double accel = u(1);

  xdot(0) = v * std::cos(yaw);
  xdot(1) = v * std::sin(yaw);
  xdot(2) = v / wheelbase_ * std::tan(steer);
  xdot(3) = accel;
}

void CarKinematic::Jacobian(const VectorXdRef& x, const VectorXdRef& u, const float t,
                            Eigen::Ref<MatrixXd> jac) {
  ALTRO_UNUSED(t);
  jac.setZero();

  double yaw = x(2);
  double v = x(3);
  double steer = u(0);

  // ∂xdot/∂x
  jac(0, 2) = -v * std::sin(yaw);
  jac(0, 3) = std::cos(yaw);
  jac(1, 2) = v * std::cos(yaw);
  jac(1, 3) = std::sin(yaw);
  jac(2, 3) = std::tan(steer) / wheelbase_;

  // ∂xdot/∂u
  jac(2, 0) = v / (wheelbase_ * std::pow(std::cos(steer), 2));  // ∂xdot2/∂u0
  jac(3, 1) = 1.0;                                              // ∂xdot3/∂u1
}

void CarKinematic::Hessian(const VectorXdRef& x, const VectorXdRef& u, const float t,
                           const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) {
  ALTRO_UNUSED(t);
  ALTRO_UNUSED(u);
  hess.setZero();

  double yaw = x(2);
  double v = x(3);

  // Only non-zero second derivatives come from xdot(0) and xdot(1)
  hess(2, 2) = -b(0) * v * std::cos(yaw) - b(1) * v * std::sin(yaw);
  hess(2, 3) = -b(0) * std::sin(yaw) + b(1) * std::cos(yaw);
  hess(3, 2) = hess(2, 3);
}

CarExtended::CarExtended(double length,
                         double width,
                         double height,
                         double wheelbase,
                         double rear_overhang)
    : length_(length),
      width_(width),
      height_(height),
      wheelbase_(wheelbase),
      rear_overhang_(rear_overhang) {
  front_overhang_ = length_ - (rear_overhang_ + wheelbase_);
}

void CarExtended::Evaluate(const VectorXdRef& x, const VectorXdRef& u, const float t,
                           Eigen::Ref<VectorXd> xdot) {
  ALTRO_UNUSED(t);

  double yaw = x(2);
  double kappa = x(3);
  double v = x(4);
  double a = x(5);

  double kappa_dot = u(0);
  double jerk = u(1);

  xdot(0) = v * std::cos(yaw);
  xdot(1) = v * std::sin(yaw);
  xdot(2) = v * kappa;
  xdot(3) = kappa_dot;
  xdot(4) = a;
  xdot(5) = jerk;
}

void CarExtended::Jacobian(const VectorXdRef& x, const VectorXdRef& u, const float t,
                           Eigen::Ref<MatrixXd> jac) {
  ALTRO_UNUSED(t);
  ALTRO_UNUSED(u);
  jac.setZero();

  double yaw = x(2);
  double kappa = x(3);
  double v = x(4);

  // ∂xdot/∂x
  jac(0, 2) = -v * std::sin(yaw);
  jac(0, 4) = std::cos(yaw);

  jac(1, 2) = v * std::cos(yaw);
  jac(1, 4) = std::sin(yaw);

  jac(2, 3) = v;  // ∂xdot2/∂kappa
  jac(2, 4) = kappa; // 
  
  jac(4, 5) = 1; // v_dot = a, so dv_dot/da = 1

  // ∂xdot/∂u
  jac(3, 0) = 1; // curvature rate control
  jac(5, 1) = 1; // longitudinal jerk control
}

void CarExtended::Hessian(const VectorXdRef& x, const VectorXdRef& u, const float t,
                          const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) {
  ALTRO_UNUSED(t);
  ALTRO_UNUSED(u);
  hess.setZero();

  double yaw = x(2);
  double v = x(4);

  // Only non-zero second derivatives come from xdot(0) and xdot(1)
  hess(2, 2) = -b(0) * v * std::cos(yaw) - b(1) * v * std::sin(yaw);
  hess(2, 4) = -b(0) * std::sin(yaw) + b(1) * std::cos(yaw);
  hess(4, 2) = hess(2, 4);
}

}  // namespace examples
}  // namespace altro
