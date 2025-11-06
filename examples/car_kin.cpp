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

}  // namespace examples
}  // namespace altro
