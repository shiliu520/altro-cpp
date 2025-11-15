// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <fmt/format.h>
#include <limits>
#include <vector>

#include "altro/common/trajectory.hpp"
#include "altro/constraints/constraint.hpp"
#include "altro/eigentypes.hpp"
#include "altro/utils/utils.hpp"

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
  int OutputDimension() const override { return xf_.size(); }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> c) override {
    ALTRO_UNUSED(u);
    c = x - xf_;
  }
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<MatrixXd> jac) override {
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

  std::string GetLabel() const override { return "Control Bound";}

  int ControlDimension() const override { return m_; }

  int OutputDimension() const override {
    return index_lower_bound_.size() + index_upper_bound_.size();
  }

  void Evaluate(const VectorXdRef& /*x*/, const VectorXdRef& u,
                Eigen::Ref<VectorXd> c) override {
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

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<MatrixXd> jac) override {
    (void) u; // surpress erroneous unused variable error
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
      : A_(G),
        B_(H),
        c_(-w),
        n_constraints_(G.rows()) {
    ALTRO_ASSERT(G.rows() == H.rows(), "G and H must have same number of rows.");
    ALTRO_ASSERT(G.rows() == w.size(), "w must have same size as number of constraints.");
    ALTRO_ASSERT(G.cols() >= 0 && H.cols() >= 0, "Invalid matrix dimensions.");
  }

  // Original constructor: lhs <= rhs  =>  (A_lhs - A_rhs)x + (B_lhs - B_rhs)u + (c_lhs - c_rhs) <= 0
  LinearStateControlConstraint(const Eigen::MatrixXd& A_lhs, const Eigen::MatrixXd& B_lhs,
                               const Eigen::VectorXd& c_lhs,
                               const Eigen::MatrixXd& A_rhs, const Eigen::MatrixXd& B_rhs,
                               const Eigen::VectorXd& c_rhs)
      : A_(A_lhs - A_rhs),
        B_(B_lhs - B_rhs),
        c_(c_lhs - c_rhs),
        n_constraints_(A_.rows()) {
    ALTRO_ASSERT(A_lhs.rows() == B_lhs.rows(), "A_lhs and B_lhs row mismatch.");
    ALTRO_ASSERT(A_rhs.rows() == B_rhs.rows(), "A_rhs and B_rhs row mismatch.");
    ALTRO_ASSERT(A_lhs.rows() == A_rhs.rows(), "LHS and RHS must have same number of constraints.");
    ALTRO_ASSERT(c_lhs.size() == c_rhs.size(), "c_lhs and c_rhs size mismatch.");
    ALTRO_ASSERT(A_lhs.cols() == A_rhs.cols(), "State dimension mismatch in A matrices.");
    ALTRO_ASSERT(B_lhs.cols() == B_rhs.cols(), "Control dimension mismatch in B matrices.");
  }

  std::string GetLabel() const override { return "Linear State-Control Constraint"; }

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

}  // namespace examples
}  // namespace altro