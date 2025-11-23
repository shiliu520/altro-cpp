// Copyright [2021] Optimus Ride Inc.

#pragma once

#include "altro/eigentypes.hpp"
#include "altro/utils/utils.hpp"
#include "examples/quadratic_cost.hpp"
#include "examples/reference_line.hpp"
#include "altro/problem/costfunction.hpp"

namespace altro {
namespace examples {

// Huber loss helper (scalar)
double HuberLoss(double z, double delta = 1.0);
double HuberLossDerivative(double z, double delta = 1.0);

// 1. Centripetal acceleration: f = (v^2 * kappa)^2
class CentripetalAccelerationCost : public problem::CostFunction {
 public:
  CentripetalAccelerationCost(double weight = 1.0, bool terminal = false);
  
  int StateDimension() const override { return 6; }
  int ControlDimension() const override { return 2; }
  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override;
  void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u,
               Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
               Eigen::Ref<MatrixXd> dudu) override;

 private:
  double weight_;
  bool terminal_;
};

// 2. Centripetal jerk: f = (2*v*a*kappa + v^2 * kappa_dot)^2
// Note: kappa_dot = u(0), a = x(5), v = x(4), kappa = x(3)
class CentripetalJerkCost : public problem::CostFunction {
 public:
  CentripetalJerkCost(double weight = 1.0, bool terminal = false);
  
  int StateDimension() const override { return 6; }
  int ControlDimension() const override { return 2; }
  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override;
  void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u,
               Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
               Eigen::Ref<MatrixXd> dudu) override;

 private:
  double weight_;
  bool terminal_;
};

// 4. Lateral distance to a given projection point
class LateralDistanceHuberCost : public problem::CostFunction {
 public:
  LateralDistanceHuberCost(const Eigen::Vector2d& proj_pos,
                           double weight = 1.0,
                           double delta = 1.0,
                           bool terminal = false);

  int StateDimension() const override { return 6; }
  int ControlDimension() const override { return 2; }

  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override;
  void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u,
               Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
               Eigen::Ref<MatrixXd> dudu) override;

 private:
  Eigen::Vector2d proj_pos_;
  double weight_;
  double delta_;
  bool terminal_;
};

class TargetSpeedHuberCost : public problem::CostFunction {
 public:
  TargetSpeedHuberCost(double weight = 1.0,
                       double v_target = 5.0,
                       double delta = 1.0,
                       bool terminal = false);

  int StateDimension() const override { return 6; }
  int ControlDimension() const override { return 2; }
  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override;
  void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u,
               Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
               Eigen::Ref<MatrixXd> dudu) override;

 private:
  double weight_;
  double v_target_;
  double delta_;
  bool terminal_;
};

// Helper: Create QuadraticCost for curvature rate (kappa_dot = u(0))
class CurvatureRateCost : public problem::CostFunction {
 public:
  CurvatureRateCost(double weight = 1.0, bool terminal = false);

  int StateDimension() const override { return 6; }
  int ControlDimension() const override { return 2; }

  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override;
  void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u,
               Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
               Eigen::Ref<MatrixXd> dudu) override;

 private:
  double weight_;
  bool terminal_;
};

class LinearJerkCost : public problem::CostFunction {
 public:
  LinearJerkCost(double weight = 1.0, bool terminal = false);

  int StateDimension() const override { return 6; }
  int ControlDimension() const override { return 2; }

  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override;
  void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u,
               Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
               Eigen::Ref<MatrixXd> dudu) override;

 private:
  double weight_;
  bool terminal_;
};

class ReferenceTrackingCost : public problem::CostFunction {
public:
    ReferenceTrackingCost(
        std::shared_ptr<ReferenceLineProjector> projector,
        double weight_lateral = 1.0,
        double weight_speed = 1.0,
        double delta_lateral = 1.0,
        double delta_speed = 1.0,
        bool terminal = false
    );

    int StateDimension() const override { return 6; }
    int ControlDimension() const override { return 2; }

    double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override;
    void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                  Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) override;
    void Hessian(const VectorXdRef& x, const VectorXdRef& u,
                 Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
                 Eigen::Ref<MatrixXd> dudu) override;

private:
    const ReferenceLine::ProjectionResult& GetProjection(const Eigen::VectorXd& x) const;

    std::shared_ptr<ReferenceLineProjector> projector_;
    double w_lat_, w_vel_;
    double delta_lat_, delta_vel_;
    bool terminal_;
};


class SumCost : public problem::CostFunction {
 public:
  explicit SumCost(const std::vector<std::shared_ptr<problem::CostFunction>>& costs);

  SumCost(const SumCost&) = delete;
  SumCost& operator=(const SumCost&) = delete;
  SumCost(SumCost&&) = default;
  SumCost& operator=(SumCost&&) = default;

  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override;

  void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) override;

  void Hessian(const VectorXdRef& x, const VectorXdRef& u,
               Eigen::Ref<MatrixXd> dxdx,
               Eigen::Ref<MatrixXd> dxdu,
               Eigen::Ref<MatrixXd> dudu) override;

  bool HasHessian() const override;

  int StateDimension() const override { return 6; }
  int ControlDimension() const override { return 2; }

 private:
  std::vector<std::shared_ptr<problem::CostFunction>> costs_;
};

}  // namespace examples
}  // namespace altro