// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <fmt/format.h>
#include <limits>
#include <vector>

#include "altro/common/trajectory.hpp"
#include "altro/constraints/constraint.hpp"
#include "altro/eigentypes.hpp"
#include "altro/utils/utils.hpp"
#include "examples/reference_line.hpp"
#include "examples/car_kin.hpp"

namespace altro {
namespace examples {

class GoalConstraint : public constraints::Constraint<constraints::Equality> {
 public:
  explicit GoalConstraint(const VectorXd& xf) : xf_(xf) {}

  static constraints::ConstraintPtr<constraints::Equality> Create(const VectorXd& xf) {
    return std::make_shared<GoalConstraint>(xf);
  }

  std::string GetLabel() const override { return "Goal Constraint"; }
  int StateDimension() const override { return xf_.size(); }
  int ControlDimension() const override { return 0; }
  int OutputDimension() const override { return xf_.size(); }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> c) override {
    ALTRO_UNUSED(u);
    c = x - xf_;
  }
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    jac.setIdentity();
  }

 private:
  VectorXd xf_;
};

class ControlBound : public constraints::Constraint<constraints::NegativeOrthant> {
 public:
  explicit ControlBound(const int m)
      : m_(m),
        lower_bound_(m, -std::numeric_limits<double>::infinity()),
        upper_bound_(m, +std::numeric_limits<double>::infinity()) {}

  ControlBound(const std::vector<double>& lb, const std::vector<double>& ub)
      : m_(lb.size()), lower_bound_(lb), upper_bound_(ub) {
    ALTRO_ASSERT(lb.size() == ub.size(), "Upper and lower bounds must have the same length.");
    ALTRO_ASSERT(lb.size() > 0, "Cannot pass in empty bounds.");
    GetFiniteIndices(upper_bound_, &index_upper_bound_);
    GetFiniteIndices(lower_bound_, &index_lower_bound_);
    ValidateBounds();
  }

  void SetUpperBound(const std::vector<double>& ub) {
    ALTRO_ASSERT(ub.size() == static_cast<size_t>(m_),
                 "Inconsistent control dimension when setting upper bound.");
    upper_bound_ = ub;
    GetFiniteIndices(upper_bound_, &index_upper_bound_);
    ValidateBounds();
  }

  void SetUpperBound(std::vector<double>&& ub) {
    ALTRO_ASSERT(ub.size() == static_cast<size_t>(m_),
                 "Inconsistent control dimension when setting upper bound.");
    upper_bound_ = std::move(ub);
    GetFiniteIndices(upper_bound_, &index_upper_bound_);
    ValidateBounds();
  }

  void SetLowerBound(const std::vector<double>& lb) {
    ALTRO_ASSERT(lb.size() == static_cast<size_t>(m_),
                 "Inconsistent control dimension when setting lower bound.");
    lower_bound_ = lb;
    GetFiniteIndices(lower_bound_, &index_lower_bound_);
    ValidateBounds();
  }

  void SetLowerBound(std::vector<double>&& lb) {
    ALTRO_ASSERT(lb.size() == static_cast<size_t>(m_),
                 "Inconsistent control dimension when setting lower bound.");
    lower_bound_ = std::move(lb);
    GetFiniteIndices(lower_bound_, &index_lower_bound_);
    ValidateBounds();
  }

  std::string GetLabel() const override { return "Control Bound"; }
  int StateDimension() const override { return 0; }
  int ControlDimension() const override { return m_; }

  int OutputDimension() const override {
    return index_lower_bound_.size() + index_upper_bound_.size();
  }

  void Evaluate(const VectorXdRef& /*x*/, const VectorXdRef& u, Eigen::Ref<VectorXd> c) override {
    ALTRO_ASSERT(u.size() == m_, "Inconsistent control dimension when evaluating control bound.");

    for (size_t i = 0; i < index_lower_bound_.size(); ++i) {
      size_t j = index_lower_bound_[i];
      c(i) = lower_bound_.at(j) - u(j);
    }
    int offset = index_lower_bound_.size();
    for (size_t i = 0; i < index_upper_bound_.size(); ++i) {
      size_t j = index_upper_bound_[i];
      c(i + offset) = u(j) - upper_bound_.at(j);
    }
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    (void)u;  // surpress erroneous unused variable error
    ALTRO_ASSERT(u.size() == m_, "Inconsistent control dimension when evaluating control bound.");
    jac.setZero();

    int n = x.size();  // state dimension
    for (size_t i = 0; i < index_lower_bound_.size(); ++i) {
      size_t j = index_lower_bound_[i];
      jac(i, n + j) = -1;
    }
    int offset = index_lower_bound_.size();
    for (size_t i = 0; i < index_upper_bound_.size(); ++i) {
      size_t j = index_upper_bound_[i];
      jac(i + offset, n + j) = 1;
    }
  }

 private:
  void ValidateBounds() {
    for (int i = 0; i < m_; ++i) {
      ALTRO_ASSERT(lower_bound_[i] <= upper_bound_[i],
                   "Lower bound isn't less than the upper bound.");
    }
  }
  static void GetFiniteIndices(const std::vector<double>& bound, std::vector<size_t>* index) {
    index->clear();
    for (size_t i = 0; i < bound.size(); ++i) {
      // if (std::abs(bound[i]) < std::numeric_limits<double>::max()) {
      if (std::isfinite(bound[i])) {
        index->emplace_back(i);
      }
    }
  }
  int m_;
  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;
  std::vector<size_t> index_lower_bound_;
  std::vector<size_t> index_upper_bound_;
};

class StateBound : public constraints::Constraint<constraints::NegativeOrthant> {
 public:
  explicit StateBound(const int n)
      : n_(n),
        lower_bound_(n, -std::numeric_limits<double>::infinity()),
        upper_bound_(n, +std::numeric_limits<double>::infinity()) {}

  StateBound(const std::vector<double>& lb, const std::vector<double>& ub)
      : n_(lb.size()), lower_bound_(lb), upper_bound_(ub) {
    ALTRO_ASSERT(lb.size() == ub.size(), "Upper and lower bounds must have the same length.");
    ALTRO_ASSERT(lb.size() > 0, "Cannot pass in empty bounds.");
    GetFiniteIndices(upper_bound_, &index_upper_bound_);
    GetFiniteIndices(lower_bound_, &index_lower_bound_);
    ValidateBounds();
  }

  void SetUpperBound(const std::vector<double>& ub) {
    ALTRO_ASSERT(ub.size() == static_cast<size_t>(n_),
                 "Inconsistent state dimension when setting upper bound.");
    upper_bound_ = ub;
    GetFiniteIndices(upper_bound_, &index_upper_bound_);
    ValidateBounds();
  }

  void SetUpperBound(std::vector<double>&& ub) {
    ALTRO_ASSERT(ub.size() == static_cast<size_t>(n_),
                 "Inconsistent state dimension when setting upper bound.");
    upper_bound_ = std::move(ub);
    GetFiniteIndices(upper_bound_, &index_upper_bound_);
    ValidateBounds();
  }

  void SetLowerBound(const std::vector<double>& lb) {
    ALTRO_ASSERT(lb.size() == static_cast<size_t>(n_),
                 "Inconsistent state dimension when setting lower bound.");
    lower_bound_ = lb;
    GetFiniteIndices(lower_bound_, &index_lower_bound_);
    ValidateBounds();
  }

  void SetLowerBound(std::vector<double>&& lb) {
    ALTRO_ASSERT(lb.size() == static_cast<size_t>(n_),
                 "Inconsistent state dimension when setting lower bound.");
    lower_bound_ = std::move(lb);
    GetFiniteIndices(lower_bound_, &index_lower_bound_);
    ValidateBounds();
  }

  std::string GetLabel() const override { return "State Bound"; }

  int StateDimension() const override { return n_; }
  int ControlDimension() const override { return 0; }

  int OutputDimension() const override {
    return index_lower_bound_.size() + index_upper_bound_.size();
  }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> c) override {
    ALTRO_ASSERT(x.size() == n_, "Inconsistent state dimension when evaluating state bound.");

    for (size_t i = 0; i < index_lower_bound_.size(); ++i) {
      size_t j = index_lower_bound_[i];
      c(i) = lower_bound_.at(j) - x(j);
    }
    int offset = index_lower_bound_.size();
    for (size_t i = 0; i < index_upper_bound_.size(); ++i) {
      size_t j = index_upper_bound_[i];
      c(i + offset) = x(j) - upper_bound_.at(j);
    }

    ALTRO_UNUSED(u);
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    (void)u;  // suppress unused variable warning
    ALTRO_ASSERT(x.size() == n_, "Inconsistent state dimension when evaluating state bound.");
    jac.setZero();

    for (size_t i = 0; i < index_lower_bound_.size(); ++i) {
      size_t j = index_lower_bound_[i];
      jac(i, j) = -1;
    }
    int offset = index_lower_bound_.size();
    for (size_t i = 0; i < index_upper_bound_.size(); ++i) {
      size_t j = index_upper_bound_[i];
      jac(i + offset, j) = 1;
    }

    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
  }

 private:
  void ValidateBounds() {
    for (int i = 0; i < n_; ++i) {
      ALTRO_ASSERT(lower_bound_[i] <= upper_bound_[i],
                   "Lower bound isn't less than the upper bound.");
    }
  }

  static void GetFiniteIndices(const std::vector<double>& bound, std::vector<size_t>* index) {
    index->clear();
    for (size_t i = 0; i < bound.size(); ++i) {
      // if (std::abs(bound[i]) < std::numeric_limits<double>::max()) {
      if (std::isfinite(bound[i])) {
        index->emplace_back(i);
      }
    }
  }

  int n_;
  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;
  std::vector<size_t> index_lower_bound_;
  std::vector<size_t> index_upper_bound_;
};

class LinearStateControlConstraint : public constraints::Constraint<constraints::NegativeOrthant> {
 public:
  // Constructor from standard affine inequality: G * x + H * u <= w
  LinearStateControlConstraint(const Eigen::MatrixXd& G, const Eigen::MatrixXd& H,
                               const Eigen::VectorXd& w)
      : A_(G), B_(H), c_(-w), n_constraints_(G.rows()) {
    ALTRO_ASSERT(G.rows() == H.rows(), "G and H must have same number of rows.");
    ALTRO_ASSERT(G.rows() == w.size(), "w must have same size as number of constraints.");
    ALTRO_ASSERT(G.cols() >= 0 && H.cols() >= 0, "Invalid matrix dimensions.");
  }

  // Original constructor: lhs <= rhs  =>  (A_lhs - A_rhs)x + (B_lhs - B_rhs)u + (c_lhs - c_rhs) <=
  // 0
  LinearStateControlConstraint(const Eigen::MatrixXd& A_lhs, const Eigen::MatrixXd& B_lhs,
                               const Eigen::VectorXd& c_lhs, const Eigen::MatrixXd& A_rhs,
                               const Eigen::MatrixXd& B_rhs, const Eigen::VectorXd& c_rhs)
      : A_(A_lhs - A_rhs), B_(B_lhs - B_rhs), c_(c_lhs - c_rhs), n_constraints_(A_.rows()) {
    ALTRO_ASSERT(A_lhs.rows() == B_lhs.rows(), "A_lhs and B_lhs row mismatch.");
    ALTRO_ASSERT(A_rhs.rows() == B_rhs.rows(), "A_rhs and B_rhs row mismatch.");
    ALTRO_ASSERT(A_lhs.rows() == A_rhs.rows(), "LHS and RHS must have same number of constraints.");
    ALTRO_ASSERT(c_lhs.size() == c_rhs.size(), "c_lhs and c_rhs size mismatch.");
    ALTRO_ASSERT(A_lhs.cols() == A_rhs.cols(), "State dimension mismatch in A matrices.");
    ALTRO_ASSERT(B_lhs.cols() == B_rhs.cols(), "Control dimension mismatch in B matrices.");
  }

  std::string GetLabel() const override { return "Linear State-Control Constraint"; }

  int StateDimension() const override { return A_.cols(); }
  int ControlDimension() const override { return B_.cols(); }
  int OutputDimension() const override { return n_constraints_; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> c) override {
    ALTRO_ASSERT(x.size() == A_.cols(), "State dimension does not match constraint matrix.");
    ALTRO_ASSERT(u.size() == B_.cols(), "Control dimension does not match constraint matrix.");
    ALTRO_ASSERT(c.size() == n_constraints_, "Output vector size mismatch.");

    c.noalias() = A_ * x + B_ * u + c_;
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    ALTRO_ASSERT(x.size() == A_.cols(), "State dimension does not match constraint matrix.");
    ALTRO_ASSERT(u.size() == B_.cols(), "Control dimension does not match constraint matrix.");
    jac.resize(n_constraints_, x.size() + u.size());
    jac.block(0, 0, n_constraints_, x.size()) = A_;
    jac.block(0, x.size(), n_constraints_, u.size()) = B_;
  }

 private:
  Eigen::MatrixXd A_;  // Combined state matrix (G or A_lhs - A_rhs)
  Eigen::MatrixXd B_;  // Combined control matrix (H or B_lhs - B_rhs)
  Eigen::VectorXd c_;  // Combined offset (-w or c_lhs - c_rhs)
  int n_constraints_;
};

// ============================================================================
// Nonlinear Constraints for CarExtended Model
// ============================================================================

class CentripetalAccelerationConstraint
    : public constraints::Constraint<constraints::NegativeOrthant> {
 public:
  explicit CentripetalAccelerationConstraint(double a_max) : a_max_(a_max) {}

  std::string GetLabel() const override { return "Centripetal Acceleration Constraint"; }

  int StateDimension() const override { return 6; }
  int ControlDimension() const override { return 0; }
  int OutputDimension() const override { return 2; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> c) override {
    ALTRO_UNUSED(u);
    double v = x(4);
    double kappa = x(3);
    double term = v * v * kappa;

    c(0) = term - a_max_;
    c(1) = -term - a_max_;
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    ALTRO_UNUSED(u);
    jac.setZero();

    double v = x(4);
    double kappa = x(3);

    double dv2k_dv = 2.0 * v * kappa;
    double dv2k_dkappa = v * v;

    jac(0, 4) = dv2k_dv;      // ∂/∂v
    jac(0, 3) = dv2k_dkappa;  // ∂/∂κ

    jac(1, 4) = -dv2k_dv;
    jac(1, 3) = -dv2k_dkappa;
  }

 private:
  double a_max_;
};

class CentripetalJerkConstraint : public constraints::Constraint<constraints::NegativeOrthant> {
 public:
  explicit CentripetalJerkConstraint(double j_max) : j_max_(j_max) {}

  std::string GetLabel() const override { return "Centripetal Jerk Constraint"; }

  int StateDimension() const override { return 6; }
  int ControlDimension() const override { return 2; }
  int OutputDimension() const override { return 2; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> c) override {
    double v = x(4);
    double a = x(5);
    double kappa = x(3);
    double kappadot = u(0);  // u[0] = κ̇

    double jc = 2.0 * v * a * kappa + v * v * kappadot;

    c(0) = jc - j_max_;
    c(1) = -jc - j_max_;
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    jac.setZero();  // 6 states + 2 controls

    double v = x(4);
    double a = x(5);
    double kappa = x(3);
    double kappadot = u(0);

    double djc_dv = 2.0 * a * kappa + 2.0 * v * kappadot;
    double djc_da = 2.0 * v * kappa;
    double djc_dkappa = 2.0 * v * a;
    double djc_dkappadot = v * v;

    // Row 0: gradient of (jc - j_max)
    jac(0, 4) = djc_dv;         // d/dv
    jac(0, 5) = djc_da;         // d/da
    jac(0, 3) = djc_dkappa;     // d/dkappa
    jac(0, 6) = djc_dkappadot;  // d/du0

    // Row 1: gradient of (-jc - j_max)
    jac(1, 4) = -djc_dv;
    jac(1, 5) = -djc_da;
    jac(1, 3) = -djc_dkappa;
    jac(1, 6) = -djc_dkappadot;
  }

 private:
  double j_max_;
};

class HeadingTrackingConstraint : public constraints::Constraint<constraints::NegativeOrthant> {
 public:
  explicit HeadingTrackingConstraint(std::shared_ptr<ReferenceLineProjector> projector,
                                     double theta_max)
      : projector_(std::move(projector)), theta_max_(theta_max) {
    if (!projector_) {
      throw std::invalid_argument("Projector must not be null.");
    }
    if (theta_max_ <= 0) {
      throw std::invalid_argument("theta_max must be positive.");
    }
  }

  std::string GetLabel() const override { return "Heading Tracking Constraint"; }

  int StateDimension() const override { return 6; }
  int ControlDimension() const override { return 0; }
  int OutputDimension() const override { return 2; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> c) override {
    ALTRO_UNUSED(u);  // u is unused but required by interface

    const auto& proj = projector_->ProjectFromState(x);
    double theta_vehicle = x(2);
    double theta_ref = proj.theta;

    // Normalize angle difference to [-π, π]
    double diff = theta_vehicle - theta_ref;
    diff = std::fmod(diff + M_PI, 2.0 * M_PI);
    if (diff < 0) diff += 2.0 * M_PI;
    diff -= M_PI;

    c(0) = diff - theta_max_;   // ≤ 0
    c(1) = -diff - theta_max_;  // ≤ 0
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(x);
    jac.setZero();  // shape: (2, 6)

    // Approximate: ignore ∂theta_ref/∂x and ∂theta_ref/∂y
    // Only derivative w.r.t. theta (index 2) is non-zero
    jac(0, 2) = 1.0;   // ∂/∂theta of (diff - theta_max)
    jac(1, 2) = -1.0;  // ∂/∂theta of (-diff - theta_max)
  }

 private:
  std::shared_ptr<ReferenceLineProjector> projector_;
  double theta_max_;
};

class PathBoundConstraint : public constraints::Constraint<constraints::NegativeOrthant> {
public:
    PathBoundConstraint(
        std::shared_ptr<ReferenceLineProjector> left_boundary_proj,
        std::shared_ptr<ReferenceLineProjector> right_boundary_proj,
        const CarExtended& vehicle_model,
        const double left_tolerance = 0.0,
        const double right_tolerance = 0.0)
        : left_boundary_(std::move(left_boundary_proj)),
          right_boundary_(std::move(right_boundary_proj)),
          vehicle_model_(vehicle_model),
          left_tol_(left_tolerance),
          right_tol_(right_tolerance) {

        if (!left_boundary_ || !right_boundary_) {
            throw std::invalid_argument("Boundary reference lines must not be null.");
        }

        // Precompute vehicle geometry in local frame (rear axle center origin)
        double front_overhang = vehicle_model_.length() - vehicle_model_.wheelbase() - vehicle_model_.rear_overhang();
        Lf_ = vehicle_model_.wheelbase() + front_overhang;  // distance from rear axle to front bumper
        Lr_ = vehicle_model_.rear_overhang();               // distance from rear axle to rear bumper (positive)
        half_width_ = vehicle_model_.width() * 0.5;
    }

    std::string GetLabel() const override { return "Path Bound Constraint"; }

    int StateDimension() const override { return 6; }
    int ControlDimension() const override { return 0; }
    int OutputDimension() const override { return 4; } // [FR, RR, FL, RL]

    void Evaluate(const VectorXdRef& x, const VectorXdRef& u,
                                       Eigen::Ref<VectorXd> c) {
      ALTRO_UNUSED(u);

      double x_l = x(0);
      double y_l = x(1);
      double theta_l = x(2);

      Eigen::Vector2d center(x_l, y_l);
      Eigen::Rotation2Dd R(theta_l);

      // Dynamic corner positions in local frame
      Eigen::Vector2d fr_local = center + R * Eigen::Vector2d(Lf_, -half_width_);
      Eigen::Vector2d rr_local = center + R * Eigen::Vector2d(-Lr_, -half_width_);
      Eigen::Vector2d fl_local = center + R * Eigen::Vector2d(Lf_, +half_width_);
      Eigen::Vector2d rl_local = center + R * Eigen::Vector2d(-Lr_, +half_width_);

      auto proj_fr = right_boundary_->ProjectFromState(fr_local);
      auto proj_rr = right_boundary_->ProjectFromState(rr_local);
      auto proj_fl = left_boundary_->ProjectFromState(fl_local);
      auto proj_rl = left_boundary_->ProjectFromState(rl_local);

      c(0) = -(proj_fr.d + right_tol_);
      c(1) = -(proj_rr.d + right_tol_);
      c(2) = proj_fl.d - left_tol_;
      c(3) = proj_rl.d - left_tol_;
    }

    void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
        ALTRO_UNUSED(u);
        jac.setZero();

        double x_l = x(0);
        double y_l = x(1);
        double theta_l = x(2);
        double cos_t = std::cos(theta_l);
        double sin_t = std::sin(theta_l);

        // Body-frame corners
        Eigen::Vector2d fr_b(Lf_, -half_width_);
        Eigen::Vector2d rr_b(-Lr_, -half_width_);
        Eigen::Vector2d fl_b(Lf_, +half_width_);
        Eigen::Vector2d rl_b(-Lr_, +half_width_);

        // Reconstruct local positions
        Eigen::Vector2d center(x_l, y_l);
        Eigen::Vector2d fr_local = center + Eigen::Rotation2Dd(theta_l) * fr_b;
        Eigen::Vector2d rr_local = center + Eigen::Rotation2Dd(theta_l) * rr_b;
        Eigen::Vector2d fl_local = center + Eigen::Rotation2Dd(theta_l) * fl_b;
        Eigen::Vector2d rl_local = center + Eigen::Rotation2Dd(theta_l) * rl_b;

        // Project to get .theta (needed for normal direction in gradient)
        auto proj_fr = right_boundary_->ProjectFromState(fr_local);
        auto proj_rr = right_boundary_->ProjectFromState(rr_local);
        auto proj_fl = left_boundary_->ProjectFromState(fl_local);
        auto proj_rl = left_boundary_->ProjectFromState(rl_local);

        // Get left normals from theta (for gradient direction)
        auto get_left_normal = [](double theta) -> Eigen::Vector2d {
            return Eigen::Vector2d(-std::sin(theta), std::cos(theta));
        };

        Eigen::Vector2d n_fr = get_left_normal(proj_fr.theta);
        Eigen::Vector2d n_rr = get_left_normal(proj_rr.theta);
        Eigen::Vector2d n_fl = get_left_normal(proj_fl.theta);
        Eigen::Vector2d n_rl = get_left_normal(proj_rl.theta);

        // Derivative of rotated point w.r.t. theta
        auto d_rot_wrt_theta = [&](const Eigen::Vector2d& p) -> Eigen::Vector2d {
            return Eigen::Vector2d(
                -sin_t * p.x() - cos_t * p.y(),
                 cos_t * p.x() - sin_t * p.y()
            );
        };

        // --- Right boundary constraints: c = -(d + tol) → ∂c/∂x = -∂d/∂x ---
        jac(0, 0) = -n_fr.x();  // ∂/∂x_l
        jac(0, 1) = -n_fr.y();  // ∂/∂y_l
        jac(0, 2) = -d_rot_wrt_theta(fr_b).dot(n_fr);

        jac(1, 0) = -n_rr.x();
        jac(1, 1) = -n_rr.y();
        jac(1, 2) = -d_rot_wrt_theta(rr_b).dot(n_rr);

        // --- Left boundary constraints: c = d - tol → ∂c/∂x = +∂d/∂x ---
        jac(2, 0) = n_fl.x();
        jac(2, 1) = n_fl.y();
        jac(2, 2) = d_rot_wrt_theta(fl_b).dot(n_fl);

        jac(3, 0) = n_rl.x();
        jac(3, 1) = n_rl.y();
        jac(3, 2) = d_rot_wrt_theta(rl_b).dot(n_rl);

        // State dimensions 3~5 (kappa, v, a) have zero gradient — already zero by setZero()
    }

private:
    std::shared_ptr<ReferenceLineProjector> left_boundary_;
    std::shared_ptr<ReferenceLineProjector> right_boundary_;
    altro::examples::CarExtended vehicle_model_;
    double left_tol_;
    double right_tol_;

    // Cached geometry for efficiency
    double Lf_ = 0.0;        // front overhang from rear axle
    double Lr_ = 0.0;        // rear overhang (positive)
    double half_width_ = 0.0;
};

}  // namespace examples
}  // namespace altro