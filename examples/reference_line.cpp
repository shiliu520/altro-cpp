#include "examples/reference_line.h"
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

    int n = static_cast<int>(trajectory_.size());
    int start = std::max(0, prev_index_hint - 5);
    int end = std::min(n - 1, prev_index_hint + 5);

    int best_i = start;
    double min_dist2 = (vehicle_pos - trajectory_[start].head<2>()).squaredNorm();

    for (int i = start + 1; i <= end; ++i) {
        double dist2 = (vehicle_pos - trajectory_[i].head<2>()).squaredNorm();
        if (dist2 < min_dist2) {
            min_dist2 = dist2;
            best_i = i;
        }
    }

    const auto& ref = trajectory_[best_i];
    ProjectionResult res;
    res.pos = ref.head<2>();
    res.theta = ref(2);
    res.vel = ref(3);
    res.next_index_hint = best_i;

    return res;
}

}  // namespace examples
}  // namespace altro