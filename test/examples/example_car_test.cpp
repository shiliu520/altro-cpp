#include <gtest/gtest.h> // 如果 ALTRO 提供了 FD 工具；否则我们自己写

#include <Eigen/Dense>
#include <cmath>
#include <libgen.h>
#include <filesystem>
#include "examples/problems/car_extended.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "test/third_party/matplotlibcpp/matplotlibcpp.h"
#pragma GCC diagnostic pop


namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

void SaveTrajectoryPlots(const double tf, const std::vector<Eigen::Vector4d>& traj,
                         const std::string& scenario_name) {
  // Assume running from build/ directory => current_path() = .../altro-cpp/build
  std::string project_source_dir = ALTRO_PROJECT_SOURCE_DIR;
  fs::path target_dir = fs::path(project_source_dir) / "Testing" / "Temporary" / "car" / scenario_name;

  fs::create_directories(target_dir);
  std::string prefix = target_dir.string() + "/" + scenario_name + "_ref_";

  // ... rest of your code unchanged ...
  const int N = static_cast<int>(traj.size()) - 1;
  std::vector<double> time(N + 1);
  for (int k = 0; k <= N; ++k) {
    time[k] = tf * static_cast<double>(k) / N;
  }

  // Plot 1: x-y
  {
    std::vector<double> x(N + 1), y(N + 1);
    for (int k = 0; k <= N; ++k) {
      x[k] = traj[k](0);
      y[k] = traj[k](1);
    }
    plt::figure_size(800, 600);
    plt::plot(x, y, "b-o");
    plt::title("Reference Path (x vs y)");
    plt::xlabel("x [m]");
    plt::ylabel("y [m]");
    plt::grid(true);
    plt::save(prefix + "x_y.png");
    //
  }

  // Plot 2: theta, v
  {
    std::vector<double> theta(N + 1), v(N + 1);
    for (int k = 0; k <= N; ++k) {
      theta[k] = traj[k](2);
      v[k] = traj[k](3);
    }

    // === Plot 2a: theta vs time (separate figure) ===
    {
      std::vector<double> theta(N + 1);
      for (int k = 0; k <= N; ++k) {
        theta[k] = traj[k](2);
      }
      plt::figure();
      plt::plot(time, theta, "r-");
      plt::title("Heading $\\theta$ over Time");
      plt::xlabel("Time [s]");
      plt::ylabel("rad");
      plt::grid(true);
      plt::save(prefix + "theta.png");
      plt::close();
    }

    // === Plot 2b: speed vs time (separate figure) ===
    {
      std::vector<double> v(N + 1);
      for (int k = 0; k <= N; ++k) {
        v[k] = traj[k](3);
      }
      plt::figure();
      plt::plot(time, v, "b-");
      plt::title("Speed $v$ over Time");
      plt::xlabel("Time [s]");
      plt::ylabel("m/s");
      plt::grid(true);
      plt::save(prefix + "speed.png");
      plt::close();
    }
    //
  }
}

std::vector<double> ComputeArcLengthFromXY(
    const std::vector<std::pair<double, double>>& points) {
    std::vector<double> s(points.size(), 0.0);
    for (size_t i = 1; i < points.size(); ++i) {
        double dx = points[i].first  - points[i-1].first;
        double dy = points[i].second - points[i-1].second;
        s[i] = s[i-1] + std::sqrt(dx*dx + dy*dy);
    }
    return s;
}

void SaveOptimizedVsReferencePlots(
    const double tf,
    const std::vector<Eigen::Vector4d>& ref_traj,
    const std::vector<Eigen::VectorXd>& x_opt,
    const std::vector<Eigen::VectorXd>& u_opt,
    const std::string& scenario_name) {

//   fs::path project_root = fs::current_path().parent_path();
  std::string project_source_dir = ALTRO_PROJECT_SOURCE_DIR;
  fs::path target_dir = fs::path(project_source_dir) / "Testing" / "Temporary" / "car" / scenario_name;

  fs::create_directories(target_dir);
  std::string prefix = target_dir.string() + "/" + scenario_name + "_";
  std::cout << "Saving plots to directory: " << target_dir << std::endl;
//   std::cout << "Total steps in x_opt: " << x_opt.size() << std::endl;

    const int N = static_cast<int>(x_opt.size()) - 1;
    std::vector<double> t_state(N + 1);
    for (int k = 0; k <= N; ++k) {
        t_state[k] = tf * static_cast<double>(k) / N;
    }

    std::vector<double> t_control(N);
    for (int k = 0; k < N; ++k) {
        t_control[k] = tf * static_cast<double>(k) / N;
    }

    // --- 1. XY Plot ---
    {
        std::vector<double> x_ref(ref_traj.size()), y_ref(ref_traj.size());
        for (size_t i = 0; i < ref_traj.size(); ++i) {
            x_ref[i] = ref_traj[i](0);
            y_ref[i] = ref_traj[i](1);
        }

        std::vector<double> x_opt_vec(x_opt.size()), y_opt_vec(x_opt.size());
        for (size_t i = 0; i < x_opt.size(); ++i) {
            x_opt_vec[i] = x_opt[i](0);
            y_opt_vec[i] = x_opt[i](1);
        }

        plt::figure();
        plt::plot(x_ref, y_ref, {{"color", "blue"}, {"linestyle", "-"}, {"label", "pos_ref"}});
        plt::plot(x_opt_vec, y_opt_vec, {{"color", "red"}, {"linestyle", "--"}, {"label", "pos_opt"}});
        plt::xlabel("x [m]");
        plt::ylabel("y [m]");
        plt::legend();
        plt::title("XY Trajectory");
        plt::grid(true);
        plt::save(prefix + "x_y.png");
    }

    // --- 2. Theta vs time ---
    {
        // --- Reference trajectory: convert to (x,y) pairs ---
        std::vector<std::pair<double, double>> ref_points;
        ref_points.reserve(ref_traj.size());
        for (const auto& p : ref_traj) {
            ref_points.emplace_back(p(0), p(1)); // x, y
        }
        std::vector<double> s_ref = ComputeArcLengthFromXY(ref_points);
        std::vector<double> theta_ref(ref_traj.size());
        for (size_t i = 0; i < ref_traj.size(); ++i) {
            theta_ref[i] = ref_traj[i](2);
        }

        std::vector<std::pair<double, double>> opt_points;
        opt_points.reserve(x_opt.size());
        for (const auto& x : x_opt) {
            opt_points.emplace_back(x(0), x(1));
        }
        std::vector<double> s_opt = ComputeArcLengthFromXY(opt_points);
        std::vector<double> theta_opt(x_opt.size());
        for (size_t i = 0; i < x_opt.size(); ++i) {
            theta_opt[i] = x_opt[i](2);
        }

        plt::figure();
        plt::plot(s_ref, theta_ref, {{"color", "blue"}, {"linestyle", "-"}, {"label", "theta_ref"}});
        plt::plot(s_opt, theta_opt, {{"color", "red"}, {"linestyle", "--"}, {"label", "theta_opt"}});
        plt::xlabel("Time [s]");
        plt::ylabel("Heading θ [rad]");
        plt::legend();
        plt::title("Heading vs Time");
        plt::grid(true);
        plt::save(prefix + "theta_t.png");
    }

    // --- 3. Velocity vs time ---
    {
        std::vector<double> v_ref(ref_traj.size());
        for (size_t i = 0; i < ref_traj.size(); ++i) {
            v_ref[i] = ref_traj[i](3);
        }

        std::vector<double> v_opt(x_opt.size());
        for (size_t i = 0; i < x_opt.size(); ++i) {
            v_opt[i] = x_opt[i](4);
        }

        plt::figure();
        plt::plot(t_state, v_ref, {{"color", "blue"}, {"linestyle", "-"}, {"label", "vel_ref"}});
        plt::plot(t_state, v_opt, {{"color", "red"}, {"linestyle", "--"}, {"label", "vel_opt"}});
        plt::xlabel("Time [s]");
        plt::ylabel("Speed v [m/s]");
        plt::legend();
        plt::title("Speed vs Time");
        plt::grid(true);
        plt::save(prefix + "vel_t.png");
    }

    // --- 4. kappa(t) ---
    {
        std::vector<double> kappa_opt(x_opt.size());
        for (size_t i = 0; i < x_opt.size(); ++i) {
            kappa_opt[i] = x_opt[i](3);
        }

        plt::figure();
        plt::plot(t_state, kappa_opt, {{"color", "black"}, {"linestyle", "-"}, {"label", "kappa_opt"}});
        plt::xlabel("Time [s]");
        plt::ylabel("Curvature κ [1/m]");
        plt::legend();
        plt::title("Curvature vs Time");
        plt::grid(true);
        plt::save(prefix + "kappa_t.png");
    }

    // --- 5. acceleration a(t) ---
    {
        std::vector<double> a_opt(x_opt.size());
        for (size_t i = 0; i < x_opt.size(); ++i) {
            a_opt[i] = x_opt[i](5);
        }

        plt::figure();
        plt::plot(t_state, a_opt, {{"color", "black"}, {"linestyle", "-"}, {"label", "a_opt"}});
        plt::xlabel("Time [s]");
        plt::ylabel("Acceleration a [m/s²]");
        plt::legend();
        plt::title("Acceleration vs Time");
        plt::grid(true);
        plt::save(prefix + "a_t.png");
    }

    // --- 6. kappa_dot (u[0]) ---
    {
        std::vector<double> kappa_dot(u_opt.size());
        for (size_t i = 0; i < u_opt.size(); ++i) {
            kappa_dot[i] = u_opt[i](0);
        }

        plt::figure();
        plt::plot(t_control, kappa_dot, {{"color", "black"}, {"linestyle", "-"}, {"label", "kappa_dot"}});
        plt::xlabel("Time [s]");
        plt::ylabel("Curvature rate dκ/dt [1/(m·s)]");
        plt::legend();
        plt::title("Curvature Rate vs Time");
        plt::grid(true);
        plt::save(prefix + "kappa_dot_t.png");
    }

    // --- 7. jerk (u[1]) ---
    {
        std::vector<double> jerk(u_opt.size());
        for (size_t i = 0; i < u_opt.size(); ++i) {
            jerk[i] = u_opt[i](1);
        }

        plt::figure();
        plt::plot(t_control, jerk, {{"color", "black"}, {"linestyle", "-"}, {"label", "jerk"}});
        plt::xlabel("Time [s]");
        plt::ylabel("Jerk [m/s³]");
        plt::legend();
        plt::title("Jerk vs Time");
        plt::grid(true);
        plt::save(prefix + "jerk_t.png");
    }
}

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
        //   const double x_ref = prob.xf(0) * static_cast<double>(k) / N;  // same as in problem setup
        //   const double y_ref = 0.0;
        //   double dx = x(0) - x_ref;
        //   double dy = x(1) - y_ref;
        //   double e = std::sqrt(dx * dx + dy * dy);  // Euclidean distance to (x_ref, 0)
        double e = x(1);

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

    // Todo: Add AL penalty for each constraint
    // for (auto& con : prob.GetInequalityConstraints()) {
    //     VectorXd c = con->Evaluate(x, u);
    //     double rho = con->GetPenalty();
    //     total_cost += 0.5 * rho * (con->Projection(c + lambda/rho).squaredNorm());
    // }

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

TEST(CarExtendedProblemTest, QuarterTurn) {
  setenv("MPLBACKEND", "Agg", 1);
  CarExtendedProblem prob;
  prob.SetScenario(CarExtendedProblem::kQuarterTurn);
  //   SaveTrajectoryPlots(prob.tf, prob.GetReferenceLine()->GetTrajectory(), "QuarterTurn");
  auto solver_al = prob.MakeALSolver();
  auto traj = *(solver_al.GetiLQRSolver().GetTrajectory());

  //   double solver_cost = solver_al.GetiLQRSolver().Cost();
  //   double expected_cost = ComputeExpectedTotalCost(prob, traj);

  solver_al.GetOptions().verbose = altro::LogLevel::kSilent;  // kDebug
  // solver_al.GetOptions().max_iterations_outer = 30;
  solver_al.GetOptions().max_iterations_inner = 150;
  solver_al.Solve();

  std::cout << "Solver status: " << altro::SolverStatusToString(solver_al.GetStatus()) << std::endl;

  EXPECT_EQ(solver_al.GetStatus(), altro::SolverStatus::kSolved);
  EXPECT_LT(solver_al.MaxViolation(), solver_al.GetOptions().constraint_tolerance);
  EXPECT_LT(solver_al.GetStats().cost_decrease.back(), solver_al.GetOptions().cost_tolerance);
  EXPECT_LT(solver_al.GetStats().gradient.back(), solver_al.GetOptions().gradient_tolerance);

  // Extract optimized trajectory as vectors
  auto traj_opt = solver_al.GetiLQRSolver().GetTrajectory();
  std::vector<Eigen::VectorXd> x_opt(prob.N + 1);
  std::vector<Eigen::VectorXd> u_opt(prob.N);  // N control steps
  for (int k = 0; k <= prob.N; ++k) {
    x_opt[k] = traj_opt->State(k);
    if (k < prob.N) {
      u_opt[k] = traj_opt->Control(k);
    }
  }

//   SaveOptimizedVsReferencePlots(prob.tf, prob.GetReferenceLine()->GetTrajectory(), x_opt, u_opt,
//                                 "QuarterTurn");
}
}

} // namespace