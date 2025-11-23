// Copyright [2021] Optimus Ride Inc.

#include <memory>
#include <vector>

#include "examples/traj_plan_cost.hpp"

namespace altro {
namespace examples {

// ---------- Huber Loss (for future use) ----------
double HuberLoss(double z, double delta) {
  if (std::abs(z) <= delta) {
    return 0.5 * z * z;
  } else {
    return delta * (std::abs(z) - 0.5 * delta);
  }
}

double HuberLossDerivative(double z, double delta) {
  if (z > delta) return delta;
  if (z < -delta) return -delta;
  return z;
}

// ---------- 1. Centripetal Acceleration Cost ----------
CentripetalAccelerationCost::CentripetalAccelerationCost(double weight, bool terminal)
    : weight_(weight), terminal_(terminal) {}

double CentripetalAccelerationCost::Evaluate(const VectorXdRef& x, const VectorXdRef& u) {
  ALTRO_UNUSED(u);
  double v = x(4);
  double kappa = x(3);
  double cent_acc = v * v * kappa;  // a_c = v^2 * κ
  return 0.5 * weight_ * cent_acc * cent_acc;
}

void CentripetalAccelerationCost::Gradient(const VectorXdRef& x, const VectorXdRef& u,
                                           Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) {
  ALTRO_UNUSED(u);
  du.setZero();

  double v = x(4);
  double kappa = x(3);
  double cent_acc = v * v * kappa;
  double grad_scalar = weight_ * cent_acc;

  dx.setZero();
  dx(3) = grad_scalar * (v * v);        // ∂/∂κ
  dx(4) = grad_scalar * (2.0 * v * kappa); // ∂/∂v
}

void CentripetalAccelerationCost::Hessian(const VectorXdRef& x, const VectorXdRef& u,
                                          Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
                                          Eigen::Ref<MatrixXd> dudu) {
  ALTRO_UNUSED(u);
  dxdu.setZero();
  dudu.setZero();
  dxdx.setZero();

  double v = x(4);
  double kappa = x(3);
  double w = weight_;

  // Second derivatives of f = 0.5 * w * (v^2 κ)^2 = 0.5 w v^4 κ^2
  double v2 = v * v;
  double v3 = v2 * v;
  double v4 = v2 * v2;

  // ∂²f/∂κ² = w * v^4
  dxdx(3, 3) = w * v4;

  // ∂²f/∂v² = w * (6 v^2 κ^2)
  dxdx(4, 4) = w * 6.0 * v2 * kappa * kappa;

  // ∂²f/∂v∂κ = w * (4 v^3 κ)
  dxdx(3, 4) = w * 4.0 * v3 * kappa;
  dxdx(4, 3) = dxdx(3, 4);
}

// ---------- 2. Centripetal Jerk Cost ----------
CentripetalJerkCost::CentripetalJerkCost(double weight, bool terminal)
    : weight_(weight), terminal_(terminal) {}

double CentripetalJerkCost::Evaluate(const VectorXdRef& x, const VectorXdRef& u) {
  double v = x(4);
  double a = x(5);
  double kappa = x(3);
  double kappa_dot = u(0);  // steering rate

  double term = 2.0 * v * a * kappa + v * v * kappa_dot;
  return 0.5 * weight_ * term * term;
}

void CentripetalJerkCost::Gradient(const VectorXdRef& x, const VectorXdRef& u,
                                   Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) {
  double v = x(4);
  double a = x(5);
  double kappa = x(3);
  double kappa_dot = u(0);

  double term = 2.0 * v * a * kappa + v * v * kappa_dot;
  double g = weight_ * term;

  dx.setZero();
  du.setZero();

  // ∂term/∂v = 2aκ + 2v κ_dot
  dx(4) = g * (2.0 * a * kappa + 2.0 * v * kappa_dot);
  // ∂term/∂a = 2v κ
  dx(5) = g * (2.0 * v * kappa);
  // ∂term/∂κ = 2v a
  dx(3) = g * (2.0 * v * a);
  // ∂term/∂kappa_dot = v^2
  du(0) = g * (v * v);
  // du(1) unchanged (jerk not involved)
}

void CentripetalJerkCost::Hessian(const VectorXdRef& x, const VectorXdRef& u,
                                  Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
                                  Eigen::Ref<MatrixXd> dudu) {
  double v = x(4);
  double a = x(5);
  double kappa = x(3);
  double kappa_dot = u(0);
  double w = weight_;

  double term = 2.0 * v * a * kappa + v * v * kappa_dot;
  
  dxdx.setZero();
  dxdu.setZero();
  dudu.setZero();

  // 1. ∇r
  VectorXd grad_r(8); // [x(6), u(2)]
  grad_r.setZero();
  grad_r(3) = 2.0 * v * a;           // ∂r/∂κ
  grad_r(4) = 2.0 * a * kappa + 2.0 * v * kappa_dot; // ∂r/∂v
  grad_r(5) = 2.0 * v * kappa;       // ∂r/∂a
  grad_r(6) = v * v;                 // ∂r/∂κ̇ (u0)

  // 2. ∇r·∇rᵀ
  MatrixXd outer_product = w * grad_r * grad_r.transpose();
  
  // 3.  r·∇²r
  MatrixXd r_hessian_r(8, 8);
  r_hessian_r.setZero();
  
  // ∇²r
  r_hessian_r(4, 4) = 2.0 * kappa_dot;  // ∂²r/∂v²
  r_hessian_r(4, 5) = 2.0 * kappa;      // ∂²r/∂v∂a
  r_hessian_r(4, 3) = 2.0 * a;          // ∂²r/∂v∂κ
  r_hessian_r(4, 6) = 2.0 * v;          // ∂²r/∂v∂κ̇
  
  r_hessian_r(5, 4) = 2.0 * kappa;      // ∂²r/∂a∂v
  r_hessian_r(5, 3) = 2.0 * v;          // ∂²r/∂a∂κ
  
  r_hessian_r(3, 4) = 2.0 * a;          // ∂²r/∂κ∂v
  r_hessian_r(3, 5) = 2.0 * v;          // ∂²r/∂κ∂a
  
  r_hessian_r(6, 4) = 2.0 * v;          // ∂²r/∂κ̇∂v
  
  r_hessian_r *= w * term;  // 乘以 r 和权重

  // 4. Hessian
  MatrixXd exact_hessian = outer_product + r_hessian_r;

  dxdx = exact_hessian.topLeftCorner(6, 6);
  dxdu = exact_hessian.topRightCorner(6, 2);
  dudu = exact_hessian.bottomRightCorner(2, 2);
}

// ---------- 3. Curvature Rate Cost (kappa_dot = u0) ----------
CurvatureRateCost::CurvatureRateCost(double weight, bool terminal)
    : weight_(weight), terminal_(terminal) {}

double CurvatureRateCost::Evaluate(const VectorXdRef& x, const VectorXdRef& u) {
  ALTRO_UNUSED(x);
  double kappa_dot = u(0);
  return 0.5 * weight_ * kappa_dot * kappa_dot;
}

void CurvatureRateCost::Gradient(const VectorXdRef& x, const VectorXdRef& u,
                                 Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) {
  ALTRO_UNUSED(x);
  dx.setZero();
  du.setZero();

  double kappa_dot = u(0);
  du(0) = weight_ * kappa_dot;  // ∂/∂u0
  // du(1) remains 0 — no dependence on jerk
}

void CurvatureRateCost::Hessian(const VectorXdRef& x, const VectorXdRef& u,
                                Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
                                Eigen::Ref<MatrixXd> dudu) {
  ALTRO_UNUSED(x); ALTRO_UNUSED(u);
  dxdx.setZero();
  dxdu.setZero();
  dudu.setZero();

  dudu(0, 0) = weight_;  // ∂²/∂u0² = weight_
  // all other entries remain zero
}

// ---------- 4. Linear Jerk Cost (jerk = u(1)) ----------
LinearJerkCost::LinearJerkCost(double weight, bool terminal)
    : weight_(weight), terminal_(terminal) {}

double LinearJerkCost::Evaluate(const VectorXdRef& x, const VectorXdRef& u) {
  ALTRO_UNUSED(x);
  double jerk = u(1);  // u = [kappa_dot, jerk]
  return 0.5 * weight_ * jerk * jerk;
}

void LinearJerkCost::Gradient(const VectorXdRef& x, const VectorXdRef& u,
                              Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) {
  ALTRO_UNUSED(x);
  dx.setZero();
  du.setZero();

  double jerk = u(1);
  du(1) = weight_ * jerk;  // ∂/∂u1
  // du(0) remains 0 — no dependence on curvature rate
}

void LinearJerkCost::Hessian(const VectorXdRef& x, const VectorXdRef& u,
                             Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
                             Eigen::Ref<MatrixXd> dudu) {
  ALTRO_UNUSED(x); ALTRO_UNUSED(u);
  dxdx.setZero();
  dxdu.setZero();
  dudu.setZero();

  dudu(1, 1) = weight_;  // ∂²/∂u1² = weight_
  // all other entries remain zero
}

// ---------- 5. Simple Lateral Distance Cost ----------
LateralDistanceHuberCost::LateralDistanceHuberCost(
    const Eigen::Vector2d& proj_pos, double weight, double delta, bool terminal)
    : proj_pos_(proj_pos), weight_(weight), delta_(delta), terminal_(terminal) {}

double LateralDistanceHuberCost::Evaluate(const VectorXdRef& x, const VectorXdRef& u) {
  ALTRO_UNUSED(u);
  Eigen::Vector2d pos(x(0), x(1));
  double dist = (pos - proj_pos_).norm();
  return weight_ * HuberLoss(dist, delta_);
}

void LateralDistanceHuberCost::Gradient(const VectorXdRef& x, const VectorXdRef& u,
                                        Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) {
  ALTRO_UNUSED(u);
  du.setZero();
  dx.setZero();

  Eigen::Vector2d pos(x(0), x(1));
  Eigen::Vector2d diff = pos - proj_pos_;
  double d = diff.norm();

  if (d < 1e-8) return; // avoid division by zero

  Eigen::Vector2d grad_d = diff / d; // ∂d/∂x, ∂d/∂y
  double scale = weight_ * HuberLossDerivative(d, delta_);

  dx(0) = scale * grad_d.x();
  dx(1) = scale * grad_d.y();
}

void LateralDistanceHuberCost::Hessian(const VectorXdRef& x, const VectorXdRef& u,
                                       Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
                                       Eigen::Ref<MatrixXd> dudu) {
  ALTRO_UNUSED(u);
  dxdu.setZero();
  dudu.setZero();
  dxdx.setZero();

  Eigen::Vector2d pos(x(0), x(1));
  Eigen::Vector2d diff = pos - proj_pos_;
  double d = diff.norm();

  if (d < 1e-8)
  {
    dxdx.block<2,2>(0,0) = weight_ * Eigen::Matrix2d::Identity();
    return;
  }

  Eigen::Vector2d unit = diff / d;
  double dhuber_dd = HuberLossDerivative(d, delta_);
  double d2huber_dd2 = (std::abs(d) <= delta_) ? 1.0 : 0.0;

  Eigen::Matrix2d I2 = Eigen::Matrix2d::Identity();
  Eigen::Matrix2d outer = unit * unit.transpose();
  Eigen::Matrix2d hess = d2huber_dd2 * outer + (dhuber_dd / d) * (I2 - outer);

  dxdx.block<2, 2>(0, 0) = weight_ * hess;
}

// ---------- 6. Target Speed Tracking with Huber Loss ----------
TargetSpeedHuberCost::TargetSpeedHuberCost(double weight, double v_target, double delta, bool terminal)
    : weight_(weight), v_target_(v_target), delta_(delta), terminal_(terminal) {}

double TargetSpeedHuberCost::Evaluate(const VectorXdRef& x, const VectorXdRef& u) {
  ALTRO_UNUSED(u);
  double v = x(4);
  double error = v - v_target_;
  return weight_ * HuberLoss(error, delta_);
}

void TargetSpeedHuberCost::Gradient(const VectorXdRef& x, const VectorXdRef& u,
                                    Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) {
  ALTRO_UNUSED(u);
  du.setZero();
  dx.setZero();

  double v = x(4);
  double error = v - v_target_;
  double dhuber_dv = HuberLossDerivative(error, delta_);

  dx(4) = weight_ * dhuber_dv;  // only dv/dx matters
}

void TargetSpeedHuberCost::Hessian(const VectorXdRef& x, const VectorXdRef& u,
                                   Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
                                   Eigen::Ref<MatrixXd> dudu) {
  ALTRO_UNUSED(u);
  dxdu.setZero();
  dudu.setZero();
  dxdx.setZero();

  double v = x(4);
  double error = v - v_target_;

  // Second derivative of Huber loss:
  //   if |error| <= delta: d²/dv² = 1
  //   else:                d²/dv² = 0
  double d2huber_dv2 = (std::abs(error) <= delta_) ? 1.0 : 0.0;

  dxdx(4, 4) = weight_ * d2huber_dv2;
}

ReferenceTrackingCost::ReferenceTrackingCost(
    std::shared_ptr<ReferenceLineProjector> projector,
    double weight_lateral,
    double weight_speed,
    double delta_lateral,
    double delta_speed,
    bool terminal)
    : projector_(std::move(projector)),
      w_lat_(weight_lateral),
      w_vel_(weight_speed),
      delta_lat_(delta_lateral),
      delta_vel_(delta_speed),
      terminal_(terminal) {}

double ReferenceTrackingCost::Evaluate(const VectorXdRef& x, const VectorXdRef& u) {
    ALTRO_UNUSED(u);
    const auto& proj = projector_->ProjectFromState(x);

    // Lateral error: perpendicular distance to path
    double dx = x[0] - proj.pos.x();
    double dy = x[1] - proj.pos.y();
    double e_lat = dx * std::sin(proj.theta) - dy * std::cos(proj.theta);

    // Speed error
    double e_vel = x[4] - proj.vel;

    double cost_lat = w_lat_ * HuberLoss(e_lat, delta_lat_);
    double cost_vel = w_vel_ * HuberLoss(e_vel, delta_vel_);

    return cost_lat + cost_vel;
}

void ReferenceTrackingCost::Gradient(const VectorXdRef& x, const VectorXdRef& u,
                                    Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) {
    ALTRO_UNUSED(u);
    du.setZero();
    dx.setZero();

    auto& proj = projector_->ProjectFromState(x);

    // --- Lateral part ---
    double dx_val = x[0] - proj.pos.x();
    double dy_val = x[1] - proj.pos.y();
    double e_lat = dx_val * std::sin(proj.theta) - dy_val * std::cos(proj.theta);
    double dhuber_dlat = HuberLossDerivative(e_lat, delta_lat_);

    // ∂e_lat/∂x = sin(θ), ∂e_lat/∂y = -cos(θ)
    dx[0] += w_lat_ * dhuber_dlat * std::sin(proj.theta);
    dx[1] += w_lat_ * dhuber_dlat * (-std::cos(proj.theta));

    // --- Speed part ---
    double e_vel = x[4] - proj.vel;
    double dhuber_dvel = HuberLossDerivative(e_vel, delta_vel_);
    dx[4] += w_vel_ * dhuber_dvel;
}

void ReferenceTrackingCost::Hessian(const VectorXdRef& x, const VectorXdRef& u,
                                    Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
                                    Eigen::Ref<MatrixXd> dudu) {
    ALTRO_UNUSED(u);
    dxdu.setZero();
    dudu.setZero();
    dxdx.setZero();

    const auto& proj = projector_->ProjectFromState(x);

    // --- Lateral Hessian (2x2 block) ---
    double sin_t = std::sin(proj.theta);
    double cos_t = std::cos(proj.theta);

    double dx_val = x[0] - proj.pos.x();
    double dy_val = x[1] - proj.pos.y();
    double e_lat = dx_val * sin_t - dy_val * cos_t;

    // double dhuber_dlat = HuberLossDerivative(e_lat, delta_lat_);
    double d2huber_dlat2 = (std::abs(e_lat) <= delta_lat_) ? 1.0 : 0.0;

    // ∂²cost/∂x² = w_lat * [ d2huber * (∂e/∂x)^2 + dhuber * ∂²e/∂x² ]
    // But e_lat is linear in x,y → ∂²e/∂x² = 0
    // So Hessian_lat = w_lat * d2huber_dlat2 * [sin_t; -cos_t] * [sin_t, -cos_t]
    Eigen::Vector2d grad_e_lat(sin_t, -cos_t);
    Eigen::Matrix2d hess_lat = w_lat_ * d2huber_dlat2 * (grad_e_lat * grad_e_lat.transpose());
    dxdx.block<2, 2>(0, 0) = hess_lat;

    // --- Speed Hessian (scalar) ---
    double e_vel = x[4] - proj.vel;
    double d2huber_dvel2 = (std::abs(e_vel) <= delta_vel_) ? 1.0 : 0.0;
    dxdx(4, 4) = w_vel_ * d2huber_dvel2;
}

SumCost::SumCost(const std::vector<std::shared_ptr<problem::CostFunction>>& costs)
    : costs_(costs) {
  if (costs_.empty()) {
    throw std::invalid_argument("SumCost: cost list is empty");
  }
}

double SumCost::Evaluate(const VectorXdRef& x, const VectorXdRef& u) {
  double total = 0.0;
  for (const auto& cost : costs_) {
    total += cost->Evaluate(x, u);
  }
  return total;
}

void SumCost::Gradient(const VectorXdRef& x, const VectorXdRef& u,
                       Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) {
  dx.setZero();
  du.setZero();

  for (const auto& cost : costs_) {
    VectorXd c_dx = VectorXd::Zero(dx.size());
    VectorXd c_du = VectorXd::Zero(du.size());
    cost->Gradient(x, u, c_dx, c_du);
    dx += c_dx;
    du += c_du;
  }
}

void SumCost::Hessian(const VectorXdRef& x, const VectorXdRef& u,
                      Eigen::Ref<MatrixXd> dxdx,
                      Eigen::Ref<MatrixXd> dxdu,
                      Eigen::Ref<MatrixXd> dudu) {
  dxdx.setZero();
  dxdu.setZero();
  dudu.setZero();

  for (const auto& cost : costs_) {
    MatrixXd c_dxdx = MatrixXd::Zero(dxdx.rows(), dxdx.cols());
    MatrixXd c_dxdu = MatrixXd::Zero(dxdu.rows(), dxdu.cols());
    MatrixXd c_dudu = MatrixXd::Zero(dudu.rows(), dudu.cols());
    cost->Hessian(x, u, c_dxdx, c_dxdu, c_dudu);
    dxdx += c_dxdx;
    dxdu += c_dxdu;
    dudu += c_dudu;
  }
}

bool SumCost::HasHessian() const {
  for (const auto& cost : costs_) {
    if (!cost->HasHessian()) {
      return false;
    }
  }
  return true;
}
}  // namespace examples
}  // namespace altro