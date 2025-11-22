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
        int next_index_hint;      // for warm-starting
    };

    explicit ReferenceLine(std::vector<Eigen::Vector4d> trajectory)
        : trajectory_(std::move(trajectory)) {}

    ProjectionResult Project(
        const Eigen::Vector2d& vehicle_pos,
        int prev_index_hint = 0) const;

private:
    std::vector<Eigen::Vector4d> trajectory_;  // 离散点序列
};

}  // namespace examples
}  // namespace altro