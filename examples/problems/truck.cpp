// Copyright [2021] Optimus Ride Inc.

#include "examples/problems/truck.hpp"

namespace altro {
namespace problems {

TruckProblem::TruckProblem() {
}

int findClosest(Eigen::Vector4d x_init, Eigen::Matrix<double, 250, 4> xref){
  int res = 0;
  double min_len = 10000000.0;
  for (int i = 0; i < 250; ++i) {
    Eigen::Vector2d dist(x_init(0) - xref(i, 0), x_init(1) - xref(i, 1));
    if (dist.norm() <= min_len){
      res = i;
      min_len = dist.norm();
    }
  }
  return res;
}

altro::problem::Problem TruckProblem::MakeProblem(const bool add_constraints, Eigen::Vector4d x_init) {
  altro::problem::Problem prob(N);

  // goal = std::make_shared<altro::examples::GoalConstraint>(xf);

  float h;
  if (scenario_ == kTurn90) {
    tf = 3.0;
    h = GetTimeStep();

    lb = {-steer_bnd};
    ub = {+steer_bnd};
    Q.diagonal().setConstant(1e-2 * h);
    R.diagonal().setConstant(1e-2 * h);
    Qf.diagonal().setConstant(100.0);

  } else if (scenario_ == kThreeObstacles) {
    tf = 5.0;
    h = GetTimeStep();

    Q.diagonal().setConstant(1.0 * h);
    R.diagonal().setConstant(0.5 * h);
    Qf.diagonal().setConstant(10.0);
    x0.setZero();
    xf << 50, 6, 0, 0;
    u0.setConstant(0.0);

    const double scaling = 50.0;
    constexpr int num_obstacles = 2;
    cx = Eigen::Vector2d(0.4, 0.6);  // x-coordinates of obstacles
    cy = Eigen::Vector2d(0.1, 0.2);  // y-coordinates of obstacles
    cr = Eigen::Vector2d::Constant(2);  // radii of obstacles
    cx *= scaling;
    cy *= scaling;

    altro::examples::CircleConstraint obs;
    for (int i = 0; i < num_obstacles; ++i) {
      obs.AddObstacle(cx(i), cy(i), cr(i));
    }
    obstacles = std::move(obs);

    lb = {-steer_bnd};
    ub = {+steer_bnd};

    for (int k = 1; k < N; ++k) {
      std::shared_ptr<altro::constraints::Constraint<altro::constraints::Inequality>> obs =
          std::make_shared<altro::examples::CircleConstraint>(obstacles);
      prob.SetConstraint(obs, k);
    }

    // Cost Function
    for (int k = 0; k < N; ++k) {
      qcost =
          std::make_shared<examples::QuadraticCost>(examples::QuadraticCost::LQRCost(Q, R, xf, uref));
      prob.SetCostFunction(qcost, k);
    }
    qterm = std::make_shared<examples::QuadraticCost>(
        examples::QuadraticCost::LQRCost(Qf, R * 0, xf, uref, true));
    prob.SetCostFunction(qterm, N);

    // Constraints
    if (add_constraints) {
      for (int k = 0; k < N; ++k) {
        prob.SetConstraint(std::make_shared<altro::examples::ControlBound>(lb, ub), k);
      }
      prob.SetConstraint(std::make_shared<examples::GoalConstraint>(xf), N);
    }
  }
  else if (scenario_ == StraightRoadWithObs) {
    tf = 5.0;
    h = GetTimeStep();

    // Q.diagonal().setConstant(1.0 * h);
    Q << 0, 0,       0, 0,
         0, 1.0 * h, 0, 0,
         0, 0, 3.0 * h, 0,
         0, 0, 0, 3.0 * h;
    // R.diagonal().setConstant(5 * h);
    R << 1000 * h, 0,
         0, 0.5 * h;
    // Qf.diagonal().setConstant(10.0);
    Qf << 0, 0,    0, 0,
          0, 1000, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;
    x0 = x_init;
    xf << 100, 0, 0, 0;
    u0 << 0.0, 10.0;
    uref << 0.0, 10.0;

    Eigen::Matrix<double, 50, 4> xref;
    for (int i = 0; i < N; ++i){
      xref(i, 0) = xf(0) / N * i;
      xref(i, 1) = 0;
      xref(i, 2) = 0;
      xref(i, 3) = 0;
    }

    constexpr int num_obstacles = 2;
    cx = Eigen::Vector2d(40, 80);  // x-coordinates of obstacles
    cy = Eigen::Vector2d(2, -2);  // y-coordinates of obstacles
    cr = Eigen::Vector2d::Constant(3);  // radii of obstacles

    altro::examples::CircleConstraint obs;
    for (int i = 0; i < num_obstacles; ++i) {
      obs.AddObstacle(cx(i), cy(i), cr(i));
    }
    obstacles = std::move(obs);

    lb = {-steer_bnd, -v_bnd};
    ub = {+steer_bnd, +v_bnd};

    for (int k = 1; k < N; ++k) {
      std::shared_ptr<altro::constraints::Constraint<altro::constraints::Inequality>> obs =
          std::make_shared<altro::examples::CircleConstraint>(obstacles);
      prob.SetConstraint(obs, k);
    }

    // Cost Function
    for (int k = 0; k < N; ++k) {
      Eigen::Vector4d vec_xref(xref(k, 0), xref(k, 1), xref(k, 2), xref(k, 3));
      qcost = std::make_shared<examples::QuadraticCost>(
          examples::QuadraticCost::LQRCost(Q, R, vec_xref, uref));
      prob.SetCostFunction(qcost, k);
    }
    qterm = std::make_shared<examples::QuadraticCost>(
        examples::QuadraticCost::LQRCost(Qf, R * 0, xf, uref, true));
    prob.SetCostFunction(qterm, N);

    // Constraints
    if (add_constraints) {
      for (int k = 0; k < N; ++k) {
        prob.SetConstraint(std::make_shared<altro::examples::ControlBound>(lb, ub), k);
      }
      // prob.SetConstraint(std::make_shared<examples::GoalConstraint>(xf), N);
    }
  }
  else if(scenario_ == LaneChange){
    tf = 5.0;
    h = GetTimeStep();

    // Q.diagonal().setConstant(1.0 * h);
    Q << 0, 0,       0, 0,
         0, 0.5 * h, 0, 0,
         0, 0, 3.0 * h, 0,
         0, 0, 0, 3.0 * h;
    // R.diagonal().setConstant(5 * h);
    R << 50000 * h, 0,
         0, 2 * h;
    // Qf.diagonal().setConstant(1000.0);
    Qf << 0, 0,    0, 0,
          0, 1000, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;
    x0 = x_init;
    xf << 50, 3.75, 0, 0;
    u0 << 0.0, 20.0;
    uref << 0.0, 20.0;
    static double tobs = 0;
    constexpr int num_obstacles = 2;
    cx = Eigen::Vector2d(0, 10 + 10 * tobs);  // x-coordinates of obstacles
    cy = Eigen::Vector2d(3.75, 3.75 / 2);  // y-coordinates of obstacles
    cr = Eigen::Vector2d::Constant(1.8);  // radii of obstacles
    tobs += 0.1;
    altro::examples::CircleConstraint obs;
    for (int i = 0; i < num_obstacles; ++i) {
      obs.AddObstacle(cx(i), cy(i), cr(i));
    }
    obstacles = std::move(obs);

    lb = {-steer_bnd, -v_bnd};
    ub = {+steer_bnd, +v_bnd};

    for (int k = 1; k < N; ++k) {
      std::shared_ptr<altro::constraints::Constraint<altro::constraints::Inequality>> obs =
          std::make_shared<altro::examples::CircleConstraint>(obstacles);
      prob.SetConstraint(obs, k);
    }

    // Cost Function
    for (int k = 0; k < N; ++k) {
      qcost = std::make_shared<examples::QuadraticCost>(
          examples::QuadraticCost::LQRCost(Q, R, xf, uref));
      prob.SetCostFunction(qcost, k);
    }
    qterm = std::make_shared<examples::QuadraticCost>(
        examples::QuadraticCost::LQRCost(Qf, R * 0, xf, uref, true));
    prob.SetCostFunction(qterm, N);

    // Constraints
    if (add_constraints) {
      for (int k = 0; k < N; ++k) {
        prob.SetConstraint(std::make_shared<altro::examples::ControlBound>(lb, ub), k);
      }
    //   prob.SetConstraint(std::make_shared<examples::GoalConstraint>(xf), N);
    }
  }
  else if (scenario_ == ForwardObs) {
    tf = 5.0;
    h = GetTimeStep();

    // Q.diagonal().setConstant(1.0 * h);
    Q << 0, 0,       0, 0,
         0, 1 * h, 0, 0,
         0, 0, 3.0 * h, 0,
         0, 0, 0, 3.0 * h;
    // R.diagonal().setConstant(5 * h);
    R << 100 * h, 0,
         0, 0.5 * h;
    // Qf.diagonal().setConstant(10.0);
    Qf << 0, 0,    0, 0,
          0, 1000, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;
    x0 = x_init;
    xf << 100, 0, 0, 0;
    u0 << 0.0, 10.0;
    uref << 0.0, 10.0;

    constexpr int num_obstacles = 1;
    cx = Eigen::Vector2d(50, 1);  // x-coordinates of obstacles
    cy = Eigen::Vector2d(1, 1);  // y-coordinates of obstacles
    cr = Eigen::Vector2d(3, 1);  // radii of obstacles

    altro::examples::CircleConstraint obs;
    for (int i = 0; i < num_obstacles; ++i) {
      obs.AddObstacle(cx(i), cy(i), cr(i));
    }
    obstacles = std::move(obs);

    lb = {-steer_bnd, -v_bnd};
    ub = {+steer_bnd, +v_bnd};

    for (int k = 1; k < N; ++k) {
      std::shared_ptr<altro::constraints::Constraint<altro::constraints::Inequality>> obs =
          std::make_shared<altro::examples::CircleConstraint>(obstacles);
      prob.SetConstraint(obs, k);
    }

    // Cost Function
    for (int k = 0; k < N; ++k) {
      qcost = std::make_shared<examples::QuadraticCost>(
          examples::QuadraticCost::LQRCost(Q, R, xf, uref));
      prob.SetCostFunction(qcost, k);
    }
    qterm = std::make_shared<examples::QuadraticCost>(
        examples::QuadraticCost::LQRCost(Qf, R * 0, xf, uref, true));
    prob.SetCostFunction(qterm, N);

    // Constraints
    if (add_constraints) {
      for (int k = 0; k < N; ++k) {
        prob.SetConstraint(std::make_shared<altro::examples::ControlBound>(lb, ub), k);
      }
    //   prob.SetConstraint(std::make_shared<examples::GoalConstraint>(xf), N);
    }
  }
  else if(scenario_ == Uturn){
    tf = 5.0;
    h = GetTimeStep();

    // Q.diagonal().setConstant(1.0 * h);
    Q << 10, 0, 0, 0,
         0, 10, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0;
    // R.diagonal().setConstant(5 * h);
    R << 500 * h, 0,
         0, 1 * h;
    // Qf.diagonal().setConstant(10.0);
    Qf << 20, 0, 0, 0,
          0, 20, 0, 0,
          0, 0, 300, 0,
          0, 0, 0, 100;
    x0 = x_init;
    xf << 0, 20, M_PI, M_PI;
    u0  << 0.0, 5.0;
    uref << 0.0, 5.0;

    Eigen::Matrix<double, 250, 4> xref;
    for (int i = 0; i < 80; ++i){
      xref(i, 0) = 0.5 * i;
      xref(i, 1) = 0;
      xref(i, 2) = 0;
      xref(i, 3) = 0;
      xref(199 - i, 0) = 0.5 * i;
      xref(199 - i, 1) = xf(1);
      xref(199 - i, 2) = M_PI;
      xref(199 - i, 3) = M_PI;
    }
    for (int i = 200; i < 250; i++)
    {
      xref(i, 0) = -0.5 * (i - 200);
      xref(i, 1) = xf(1);
      xref(i, 2) = M_PI;
      xref(i, 3) = M_PI;
    }

    for (int i = 80; i < 120; i++)    {
      xref(i, 0) = 40;
      xref(i, 1) = xf(1) / 40 * (i - 80);
      xref(i, 2) = M_PI / 2;
      xref(i, 3) = M_PI / 2;
    }
    // for (int i = 70; i < 130; i++)    {
    //   xref(i, 0) = 35 + xf(1) / 2 * sin(M_PI * (i - 70) / 59);
    //   xref(i, 1) = 10 - xf(1) / 2 * cos(M_PI * (i - 70) / 59);
    //   xref(i, 2) = M_PI * (i - 70) / 59;
    //   xref(i, 3) = xref(i - 10, 2);
    // }
    int start_id = findClosest(x_init, xref);
    static int final = 50;
    if (final == 80 || final == 120) final += 15;
    xf = Eigen::Vector4d(xref(final, 0), xref(final, 1), xref(final, 2), xref(final, 3));
    final += 1;
    lb = {-steer_bnd, -v_bnd};
    ub = {+steer_bnd, +v_bnd};
    // std::cout << xref << std::endl;

    // Cost Function
    for (int k = 0; k < N; ++k) {
      Eigen::Vector4d vec_xref(xref(k + start_id, 0), xref(k + start_id, 1), xref(k + start_id, 2), xref(k + start_id, 3));
      qcost = std::make_shared<examples::QuadraticCost>(
          examples::QuadraticCost::LQRCost(Q*0, R, vec_xref, uref));
      prob.SetCostFunction(qcost, k);
    }
    qterm = std::make_shared<examples::QuadraticCost>(
        examples::QuadraticCost::LQRCost(Qf, R * 0, xf, uref, true));
    prob.SetCostFunction(qterm, N);
    // std::cout << 11 << std::endl;

    // Constraints
    if (add_constraints) {
      for (int k = 0; k < N; ++k) {
        prob.SetConstraint(std::make_shared<altro::examples::ControlBound>(lb, ub), k);
      }
    }
  }
else if(scenario_ == Largecurve){  // 假设 scenario_ 是枚举，已改为小写
    tf = 5.0;
    h = GetTimeStep();
    const int NStates = 4;  // 明确状态维度为4（与Truck模型一致）
    // const int N = 250;      // 预测步长（与错误中的250行对应）

    // Q：4×4 状态权重矩阵
    Q << 10, 0, 0, 0,
         0, 10, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0;
    // R：2×2 控制权重矩阵（不变，因为控制输入是2维：steer、v）
    R << 300 * h, 0,
         0, 1 * h;
    // Qf：4×4 终端状态权重矩阵
    Qf << 100, 0, 0, 0,
          0, 100, 0, 0,
          0, 0, 200, 0,
          0, 0, 0, 200;

    x0 = x_init;  // 确保 x_init 是4维向量
    xf << 0, 40, M_PI, M_PI;  // 4维目标状态

    u0 << 0.0, 10.0;
    uref << 0.0, 10.0;

    // xref：400行×4列（参考轨迹，列数=状态维度）
    Eigen::Matrix<double, 250, 4> xref;
    for (int i = 0; i < 50; ++i) {
      // 前直线段（i=0~49）：沿x轴正方向
      xref(i, 0) = 0.8 * i;              // x: 0→39.2（步长0.8，适配250点总长度）
      xref(i, 1) = 0.0;                  // y=0
      xref(i, 2) = 0.0;                  // yaw=0（朝向x轴正）
      xref(i, 3) = 0.0;                  // hitch=0

      // 后直线段（total_points-1 -i = 249~200）：沿x轴负方向（与前直线对称）
      int reverse_i = 250 - 1 - i;
      xref(reverse_i, 0) = 0.8 * i;      // x: 39.2→0（步长-0.8）
      xref(reverse_i, 1) = 40;         // y=40（与前直线y=0对称）
      xref(reverse_i, 2) = M_PI;         // yaw=π（朝向x轴负）
      xref(reverse_i, 3) = M_PI;         // hitch=π
    }

    // 第二个for循环：处理半圆曲线段（i=50~199，共150个点）
    for (int i = 50; i < 200; ++i) {
      int curve_idx = i - 50;   // 曲线段内部索引（0~149）
      double angle = M_PI * curve_idx / (200 - 50 - 1);  // 0→π（180°）
      double radius = 40 / 2;          // 半圆半径=20（与原一致）
      double center_x = 40.0;            // 半圆圆心x（衔接前直线末端）
      double center_y = 30.0;            // 半圆圆心y（与原一致）

      // x/y坐标：半圆轨迹（右→上→左转弯）
      xref(i, 0) = center_x + radius * sin(angle);  // x:40→60→40
      xref(i, 1) = center_y - radius * cos(angle);  // y:30→50→30
      xref(i, 2) = angle;                           // yaw角：0→π（均匀转向）
      xref(i, 3) = xref(std::max(i - 7, 50), 2);  // hitch角滞后7步（避免越界）
    }

    // 修正 start_id 范围，避免索引越界
    int start_id = findClosest(x_init, xref);
    start_id = std::min(start_id, static_cast<int>(xref.rows()) - N);
    start_id = std::max(start_id, 0);

    static int final = 50;
    xf << xref(final, 0), xref(final, 1), xref(final, 2), xref(final, 3);  // 4维
    final += 1;

    lb = {-steer_bnd, -v_bnd};
    ub = {+steer_bnd, +v_bnd};

    // Cost Function
    for (int k = 0; k < N; ++k) {
      Eigen::VectorXd vec_xref(NStates);
      vec_xref << xref(k + start_id, 0), xref(k + start_id, 1),
                  xref(k + start_id, 2), xref(k + start_id, 3);  // 4维参考状态
      qcost = std::make_shared<examples::QuadraticCost>(
          examples::QuadraticCost::LQRCost(Q, R, vec_xref, uref));
      prob.SetCostFunction(qcost, k);
    }
    qterm = std::make_shared<examples::QuadraticCost>(
        examples::QuadraticCost::LQRCost(Qf, R * 0, xf, uref, true));
    prob.SetCostFunction(qterm, N);

    // Constraints
    if (add_constraints) {
      for (int k = 0; k < N; ++k) {
        prob.SetConstraint(std::make_shared<altro::examples::ControlBound>(lb, ub), k);
      }
    }
  }
//   else if(scenario_ == Largecurve){
//     tf = 5.0;
//     h = GetTimeStep();
//     // Q.diagonal().setConstant(1.0 * h);
//     Q << 10, 0, 0, 0, 0, 0,
//          0, 10, 0, 0, 0, 0,
//          0, 0, 0, 0, 0, 0,
//          0, 0, 0, 0, 0, 0,
//          0, 0, 0, 0, 100, 0,
//          0, 0, 0, 0, 0, 100;
//     // R.diagonal().setConstant(5 * h);
//     R << 300 * h, 0,
//          0, 1 * h;
//     // Qf.diagonal().setConstant(10.0);
//     Qf << 100, 0, 0, 0, 0, 0,
//           0, 100, 0, 0, 0, 0,
//           0, 0, 200, 0, 0, 0,
//           0, 0, 0, 200, 0, 0,
//           0, 0, 0, 0, 100, 0,
//           0, 0, 0, 0, 0, 100;
//     x0 = x_init;
//     xf << 0, 40, M_PI, M_PI, 0, 40;
//     u0  << 0.0, 10.0;
//     uref << 0.0, 10.0;
//     Eigen::Matrix<double, 400, 6> xref;
//     for (int i = 0; i < 100; ++i){
//       xref(i, 0) = 0.5 * i;
//       xref(i, 1) = 0;
//       xref(i, 2) = 0;
//       xref(i, 3) = 0;
//       xref(i, 4) = 0.5 * i - 5;
//       xref(i, 5) = 0;
//       xref(399 - i, 0) = 0.5 * i;
//       xref(399 - i, 1) = xf(1);
//       xref(399 - i, 2) = M_PI;
//       xref(399 - i, 3) = M_PI;
//       xref(399 - i, 4) = 0.5 * i + 5;
//       xref(399 - i, 5) = 0.5 * i + 5;
//     }
//     for (int i = 100; i < 300; i++)    {
//       xref(i, 0) = 50 + xf(1) / 2 * sin(M_PI * (i - 100) / 199);
//       xref(i, 1) = 30 - xf(1) / 2 * cos(M_PI * (i - 100) / 199);
//       xref(i, 2) = M_PI * (i - 100) / 199;
//       xref(i, 3) = xref(i - 10, 2);
//       xref(i, 4) = xref(i, 0) - 5 * cos(xref(i, 3));
//       xref(i, 5) = xref(i, 1) - 5 * sin(xref(i, 3));
//     }
//     int start_id = findClosest(x_init, xref);
//     static int final = 50;
//     xf << xref(final, 0), xref(final, 1), xref(final, 2), xref(final, 3), xref(final, 4),
//         xref(final, 5);
//     final += 1;
//     lb = {-steer_bnd, -v_bnd};
//     ub = {+steer_bnd, +v_bnd};
//     // Cost Function
//     for (int k = 0; k < N; ++k) {
//     //   Eigen::Vector4d vec_xref(xref(k + start_id, 0), xref(k + start_id, 1), xref(k + start_id, 2),
//     //                            xref(k + start_id, 3), xref(k + start_id, 4), xref(k + start_id, 5));
//       Eigen::VectorXd vec_xref(NStates);
//       vec_xref << xref(k + start_id, 0), xref(k + start_id, 1), xref(k + start_id, 2),
//           xref(k + start_id, 3), xref(k + start_id, 4), xref(k + start_id, 5);
//       qcost = std::make_shared<examples::QuadraticCost>(
//           examples::QuadraticCost::LQRCost(Q, R, vec_xref, uref));
//       prob.SetCostFunction(qcost, k);
//     }
//     qterm = std::make_shared<examples::QuadraticCost>(
//         examples::QuadraticCost::LQRCost(Qf, R * 0, xf, uref, true));
//     prob.SetCostFunction(qterm, N);
//     // Constraints
//     if (add_constraints) {
//       for (int k = 0; k < N; ++k) {
//         prob.SetConstraint(std::make_shared<altro::examples::ControlBound>(lb, ub), k);
//       }
//     }
//   }

  // Dynamics
  for (int k = 0; k < N; ++k) {
    prob.SetDynamics(std::make_shared<ModelType>(model), k);
  }

  // Initial State and target state
  prob.SetInitialState(x0);

  return prob;
}

}  // namespace problems
}  // namespace altro