#include <gtest/gtest.h>
#include "examples/problems/car_extended.hpp" // 如果 ALTRO 提供了 FD 工具；否则我们自己写

#include <Eigen/Dense>
#include <cmath>

namespace {

using namespace altro;
using namespace problems;

// =============================================================================
// 1. Numerical Gradient — accepts any callable: f(z) -> scalar
// =============================================================================
template<typename Callable, typename VecType>
Eigen::VectorXd NumericalGradient(const Callable& f, const VecType& z, double eps = 1e-6) {
    VecType z_pert = z;
    Eigen::VectorXd grad(z.size());
    double c0 = f(z);  // ← 直接调用 f(z)，不是 f.Evaluate(z)
    for (int i = 0; i < static_cast<int>(z.size()); ++i) {
        z_pert[i] += eps;
        double c1 = f(z_pert);
        grad[i] = (c1 - c0) / eps;
        z_pert[i] = z[i];
    }
    return grad;
}

// =============================================================================
// 2. Numerical Hessian — also uses callable f(z)
// =============================================================================
template<typename Callable, typename VecType>
Eigen::MatrixXd NumericalHessian(const Callable& f, const VecType& z, double eps = 1e-5) {
    const int n = z.size();
    Eigen::MatrixXd hess(n, n);
    Eigen::VectorXd grad0 = NumericalGradient(f, z, eps);
    VecType z_pert = z;
    for (int i = 0; i < n; ++i) {
        z_pert[i] += eps;
        Eigen::VectorXd grad1 = NumericalGradient(f, z_pert, eps);
        hess.col(i) = (grad1 - grad0) / eps;
        z_pert[i] = z[i];
    }
    // Symmetrize to reduce numerical asymmetry
    return 0.5 * (hess + hess.transpose());
}

template<typename Callable, typename VecType>
Eigen::MatrixXd NumericalHessianCentral(const Callable& f, const VecType& x, double eps = 1e-4) {
    const int n = x.size();
    Eigen::MatrixXd H(n, n);
    VecType x_pert = x;

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            // f(x + ei*eps + ej*eps)
            x_pert = x;
            x_pert[i] += eps;
            x_pert[j] += eps;
            double f_pp = f(x_pert);

            // f(x - ei*eps - ej*eps)
            x_pert = x;
            x_pert[i] -= eps;
            x_pert[j] -= eps;
            double f_mm = f(x_pert);

            if (i == j) {
                // Diagonal: (f(x+eps*e_i) - 2f(x) + f(x-eps*e_i)) / eps^2
                x_pert = x;
                x_pert[i] += eps;
                double f_p = f(x_pert);
                x_pert = x;
                x_pert[i] -= eps;
                double f_m = f(x_pert);
                H(i, i) = (f_p - 2.0 * f(x) + f_m) / (eps * eps);
            } else {
                // Off-diagonal: (f_pp - f_pm - f_mp + f_mm) / (4*eps^2)
                x_pert = x;
                x_pert[i] += eps;
                x_pert[j] -= eps;
                double f_pm = f(x_pert);

                x_pert = x;
                x_pert[i] -= eps;
                x_pert[j] += eps;
                double f_mp = f(x_pert);

                H(i, j) = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps);
                H(j, i) = H(i, j);
            }
        }
    }
    return H;
}

// =============================================================================
// 3. calc expect cost
// =============================================================================
static double ComputeExpectedTotalCost(const altro::problems::CarExtendedProblem& prob,
                                       const altro::problems::CarExtendedProblem::TrajType& traj) {
    const int N = prob.N;
    const double h = prob.GetTimeStep();
    const int nx = altro::problems::CarExtendedProblem::NStates;   // 6
    // const int nu = altro::problems::CarExtendedProblem::NControls; // 2

    double total_cost = 0.0;

    // --- Stage costs: k = 0 to N-1 ---
    for (int k = 0; k < N; ++k) {
        const Eigen::VectorXd& x = traj.State(k);   // [x, y, θ, κ, v, a]
        const Eigen::VectorXd& u = traj.Control(k); // [κ̇, jerk]

        double cost_k = 0.0;

        // 1. Curvature rate: 0.5 * w * h * u0^2
        if (prob.w_curv_rate > 0) {
            cost_k += 0.5 * prob.w_curv_rate * h * u(0) * u(0);
        }

        // 2. Linear jerk: 0.5 * w * h * u1^2
        if (prob.w_jerk > 0) {
            cost_k += 0.5 * prob.w_jerk * h * u(1) * u(1);
        }

        // 3. Centripetal jerk: 0.5 * w * h * (2*v*a*κ + v^2*κ̇)^2
        if (prob.w_centripetal_jerk > 0) {
            double v = x(4);
            double a = x(5);
            double kappa = x(3);
            double kappadot = u(0);

            double j_c = 2.0 * v * a * kappa + v * v * kappadot;
            cost_k += 0.5 * prob.w_centripetal_jerk * h * j_c * j_c;
        }

        // 4. Centripetal acceleration: 0.5 * w * h * (v^2 * κ)^2
        if (prob.w_centric_acc > 0) {
            double v = x(4), kappa = x(3);
            double ca = v * v * kappa;
            cost_k += 0.5 * prob.w_centric_acc * h * ca * ca;
        }

        // 5. Target speed Huber loss (only in kGtest)
        if (prob.w_target_speed > 0 && prob.GetScenario() == altro::problems::CarExtendedProblem::kGtest) {
            double v = x(4);
            double v_ref = prob.xf(4);
            double e = v - v_ref;
            double d = prob.delta_speed;
            double huber = (std::abs(e) <= d) ? (0.5 * e * e) : (d * (std::abs(e) - 0.5 * d));
            cost_k += prob.w_target_speed * h * huber;
        }

        // 6. Lateral distance Huber loss to y=0 (only in kGtest)
        if (prob.w_lateral > 0
            && prob.GetScenario() == altro::problems::CarExtendedProblem::kGtest) {
          const double x_ref = prob.xf(0) * static_cast<double>(k) / N;  // same as in problem setup
          const double y_ref = 0.0;
          double dx = x(0) - x_ref;
          double dy = x(1) - y_ref;
          double e = std::sqrt(dx * dx + dy * dy);  // Euclidean distance to (x_ref, 0)

          double d = prob.delta_lateral;
          double huber = (std::abs(e) <= d) ? (0.5 * e * e) : (d * (std::abs(e) - 0.5 * d));
          cost_k += prob.w_lateral * h * huber;
        }

        total_cost += cost_k;
    }

    // --- Terminal cost: k = N ---
    if (prob.w_terminal_state > 0) {
        Eigen::VectorXd dx = traj.State(N) - prob.xf;
        Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(nx, nx) * (prob.w_terminal_state * 10.0);
        total_cost += 0.5 * dx.transpose() * Qf * dx;
    }

    return total_cost;
}

// =============================================================================
// Google Test Cases
// =============================================================================
namespace {

using namespace altro::problems;

TEST(CarExtendedProblemTest, TotalCostMatchesManualComputation) {
    CarExtendedProblem prob;
    prob.SetScenario(CarExtendedProblem::kGtest);

    prob.w_curv_rate = 0.1;
    prob.w_jerk = 0.1;
    prob.w_centripetal_jerk = 0.01;
    prob.w_centric_acc = 0.05;
    prob.w_target_speed = 1.0;
    prob.w_lateral = 1.0;
    prob.w_terminal_state = 10.0;

    auto solver = prob.MakeALSolver();
    auto traj = *(solver.GetiLQRSolver().GetTrajectory());

    double solver_cost = solver.GetiLQRSolver().Cost();
    double expected_cost = ComputeExpectedTotalCost(prob, traj);

    EXPECT_NEAR(solver_cost, expected_cost, 1e-8)
        << "Total cost from solver does not match manually computed expected cost!";
}

TEST(CarExtendedProblemTest, CostGradientsAndHessiansAreCorrect) {
    CarExtendedProblem prob;
    prob.SetScenario(CarExtendedProblem::kGtest);

    prob.w_curv_rate = 0.1;
    prob.w_jerk = 0.1;
    prob.w_centripetal_jerk = 0.01;
    prob.w_centric_acc = 0.05;
    prob.w_target_speed = 1.0;
    prob.w_lateral = 1.0;
    prob.w_terminal_state = 10.0;

    auto solver = prob.MakeALSolver();
    auto traj = solver.GetiLQRSolver().GetTrajectory();

    const int N = prob.N;
    const int nx = CarExtendedProblem::NStates;
    const int nu = CarExtendedProblem::NControls;

    auto& ilqr = solver.GetiLQRSolver();

    for (int k = 0; k <= N; ++k) {
        auto& kpf = ilqr.GetKnotPointFunction(k);
        auto cost_func = kpf.GetCostFunPtr();
        ASSERT_NE(cost_func, nullptr);

        Eigen::VectorXd x = traj->State(k);
        Eigen::VectorXd u = (k < N) ? traj->Control(k) : Eigen::VectorXd::Zero(nu);

        kpf.CalcCostExpansion(x, u);
        const auto& exp = kpf.GetCostExpansion();

        auto eval_cost = [&](const Eigen::VectorXd& z) -> double {
            if (k == N) {
                // Terminal: z == x
                return cost_func->Evaluate(z, Eigen::VectorXd::Zero(nu));
            } else {
                // Running: z = [x; u]
                return cost_func->Evaluate(z.head(nx), z.tail(nu));
            }
        };

        Eigen::VectorXd z;
        if (k == N) {
            z = x;
        } else {
            z.resize(nx + nu);
            z.head(nx) = x;
            z.tail(nu) = u;
        }

        Eigen::VectorXd grad_num = NumericalGradient(eval_cost, z, 1e-6);
        Eigen::MatrixXd hess_num = NumericalHessian(eval_cost, z, 1e-5);

        Eigen::VectorXd grad_ana;
        Eigen::MatrixXd hess_ana;

        if (k < N) {
            grad_ana.resize(nx + nu);
            grad_ana << exp.dx(), exp.du();

            hess_ana.setZero(nx + nu, nx + nu);
            hess_ana.topLeftCorner(nx, nx) = exp.dxdx();
            hess_ana.topRightCorner(nx, nu) = exp.dxdu();
            hess_ana.bottomLeftCorner(nu, nx) = exp.dxdu().transpose();
            hess_ana.bottomRightCorner(nu, nu) = exp.dudu();
        } else {
            grad_ana = exp.dx();
            hess_ana = exp.dxdx();
            // std::cout << std::setprecision(12);
            // std::cout << "Terminal k =" << k << std::endl;
            // std::cout << "x " << x.transpose() << std::endl;
            // std::cout << "u " << u.transpose() << std::endl;
            // std::cout << "xf " << prob.xf.transpose() << std::endl;
            // std::cout << "hess_ana:\n" << hess_ana << std::endl;
            // std::cout << "hess_num:\n" << hess_num << std::endl;
            // std::cout << "terminal cost: " << eval_cost(z) << std::endl;
        }

        EXPECT_LT((grad_ana - grad_num).norm(), 2e-4) << "Gradient mismatch at knot " << k;
        if (k < N) {
          EXPECT_LT((hess_ana - hess_num).norm(), 3e-5) << "Hessian mismatch at knot " << k;
        }
    }
}

}

} // namespace