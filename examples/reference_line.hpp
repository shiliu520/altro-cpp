#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace altro {
namespace examples {

class ReferenceLine {
public:
    struct ProjectionResult {
        Eigen::Vector2d pos;      // (x_ref, y_ref)
        double theta;             // path tangent angle
        double vel;               // desired speed
        double d;                 // signed lateral distance:
        int next_index_hint;      // for warm-starting
    };

    explicit ReferenceLine(std::vector<Eigen::Vector4d> trajectory)
        : trajectory_(std::move(trajectory)) {}

    ProjectionResult Project(
        const Eigen::Vector2d& vehicle_pos,
        int prev_index_hint = 0) const;

    const std::vector<Eigen::Vector4d>& GetTrajectory() const {
        return trajectory_;
    }

private:
    std::vector<Eigen::Vector4d> trajectory_;  // 离散点序列
};

class ReferenceLineProjector {
 public:
  explicit ReferenceLineProjector(std::shared_ptr<const ReferenceLine> ref_line)
      : ref_line_(std::move(ref_line)) {}

  const ReferenceLine::ProjectionResult& ProjectFromState(const Eigen::VectorXd& x) {
    ALTRO_ASSERT(x.size() >= 3, "x.size() must be at least 3.");

    Eigen::Vector2d pos(x[0], x[1]);

    if (cache_valid_) {
      double dist = (pos - last_pos_).norm();
      double heading_diff = abs(NormalizeAngle(x[2] - last_heading_));
      if (dist > max_search_radius_ || heading_diff > max_heading_change_) {
        Reset();
      }
    }

    if (!cache_valid_ || !pos.isApprox(last_pos_, 1e-9)) {
      last_pos_ = pos;
      last_heading_ = x[2];
      cached_result_ = ref_line_->Project(pos, prev_index_hint_);
      prev_index_hint_ = cached_result_.next_index_hint;
      cache_valid_ = true;
    }
    return cached_result_;
  }

  std::shared_ptr<const ReferenceLine> GetReferenceLine() const { return ref_line_; }

  void Reset() {
    cache_valid_ = false;
    prev_index_hint_ = 0;
  }

 private:
  std::shared_ptr<const ReferenceLine> ref_line_;
  bool cache_valid_ = false;
  Eigen::Vector2d last_pos_;
  double last_heading_;
  ReferenceLine::ProjectionResult cached_result_;
  int prev_index_hint_ = 0;
  double max_search_radius_ = 6.0;   // meters
  double max_heading_change_ = 0.2;  // radians
};

}  // namespace examples
}  // namespace altro