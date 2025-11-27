#include "altro/utils/utils.hpp"
#include "examples/reference_line.hpp"

#include <limits>
#include <algorithm>

namespace altro {
namespace examples {

ReferenceLine::ProjectionResult ReferenceLine::Project(
    const Eigen::Vector2d& vehicle_pos,
    int prev_index_hint) const {

    if (trajectory_.empty()) {
        throw std::runtime_error("Reference trajectory is empty!");
    }

    const size_t n = trajectory_.size();

    if (n == 1) {
        const auto& p = trajectory_[0];
        return ProjectionResult{
            p.head<2>(),
            p(2),
            p(3),
            0,
            0
        };
    }

    prev_index_hint = safe_clamp(prev_index_hint, 0, static_cast<int>(n) - 1);

    const int COARSE_WINDOW = 30;
    int coarse_start = std::max(0, prev_index_hint - COARSE_WINDOW);
    int coarse_end = std::min(static_cast<int>(n) - 1, prev_index_hint + COARSE_WINDOW);

    int best_point_idx = coarse_start;
    double min_dist2 = (vehicle_pos - trajectory_[coarse_start].head<2>()).squaredNorm();

    for (int i = coarse_start + 1; i <= coarse_end; ++i) {
        double d2 = (vehicle_pos - trajectory_[i].head<2>()).squaredNorm();
        if (d2 < min_dist2) {
            min_dist2 = d2;
            best_point_idx = i;
        }
    }

    const int FINE_WINDOW = 15; // ( ≥ COARSE_WINDOW/2）
    int seg_start = std::max(0, best_point_idx - FINE_WINDOW);
    int seg_end = std::min(static_cast<int>(n) - 2, best_point_idx + FINE_WINDOW);
    double global_min_dist2 = std::numeric_limits<double>::max();
    Eigen::Vector2d best_proj_point;
    double best_theta = 0.0;
    double best_vel = 0.0;
    int best_seg_start = seg_start;

    best_proj_point.setZero();

    seg_start = 0;
    seg_end = n - 2;
    for (int i = seg_start; i <= seg_end; ++i) {
        const Eigen::Vector2d A = trajectory_[i].head<2>();
        const Eigen::Vector2d B = trajectory_[i + 1].head<2>();

        Eigen::Vector2d AB = B - A;
        double ab2 = AB.squaredNorm();

        if (ab2 < 1e-12) {
            double dist2 = (vehicle_pos - A).squaredNorm();
            if (dist2 < global_min_dist2) {
                global_min_dist2 = dist2;
                best_proj_point = A;
                best_theta = trajectory_[i](2);
                best_vel = trajectory_[i](3);
                best_seg_start = i;
            }
            continue;
        }

        double t = (vehicle_pos - A).dot(AB) / ab2;
        t = safe_clamp(t, 0.0, 1.0);

        Eigen::Vector2d proj = A + t * AB;
        double dist2 = (vehicle_pos - proj).squaredNorm();

        if (dist2 < global_min_dist2) {
            global_min_dist2 = dist2;
            best_proj_point = proj;

            double theta_A = trajectory_[i](2);
            double theta_B = trajectory_[i + 1](2);
            best_theta = InterpolateAngle(theta_A, theta_B, t);

            double vel_A = trajectory_[i](3);
            double vel_B = trajectory_[i + 1](3);
            best_vel = vel_A + t * (vel_B - vel_A);

            best_seg_start = i;
        }
    }

    Eigen::Vector2d vec_to_vehicle = vehicle_pos - best_proj_point;
    Eigen::Vector2d left_normal(-std::sin(best_theta), std::cos(best_theta)); // 左手法向量
    double d = vec_to_vehicle.dot(left_normal);

    return ProjectionResult{
        best_proj_point,
        best_theta,
        best_vel,
        d,
        best_seg_start  // 下次从该线段开始搜索
    };
}

}  // namespace examples
}  // namespace altro