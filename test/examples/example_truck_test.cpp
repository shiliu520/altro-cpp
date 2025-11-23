// Copyright [2021] Optimus Ride Inc.

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>   // 用于 system() 调用（创建目录）
#include <cstring>   // 用于字符串处理
#include <unistd.h>  // 用于 realpath()（解析绝对路径）
#include <libgen.h>  // 用于 dirname()（提取目录路径）
#include <filesystem>

#include "altro/augmented_lagrangian/al_solver.hpp"
#include "altro/common/solver_options.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "examples/problems/truck.hpp"
#include "../third_party/matplotlibcpp/matplotlibcpp.h"

namespace fs = std::filesystem;

// 工具函数1：解析当前 .cpp 文件的所在目录（绝对路径）
std::string get_current_file_dir() {
    char current_file[PATH_MAX] = {0};
    // 1. 解析 __FILE__ 为绝对路径（处理相对路径、符号链接）
    if (realpath(__FILE__, current_file) == NULL) {
        // 如果失败，打印错误信息并终止程序
        std::cerr << "Error: Failed to resolve path for file '" << __FILE__
                  << "'. Reason: " << strerror(errno) << std::endl;
        exit(EXIT_FAILURE); // 或者根据你的需求抛出异常
    }

    // 2. 提取目录部分（dirname 会修改原字符串，所以用临时变量）
    char* dir = dirname(current_file);
    // 3. 转为 C++ string 返回
    return std::string(dir);
}

// 工具函数2：递归创建多级目录（兼容 Linux/macOS，用系统命令 mkdir -p）
bool create_directory_recursive(const std::string& dir_path) {
    // 拼接系统命令：mkdir -p 目录（-p 表示递归创建，不存在则创建，存在则不报错）
    std::string cmd = "mkdir -p \"" + dir_path + "\"";
    // 执行命令，返回 0 表示成功
    int ret = system(cmd.c_str());
    return ret == 0;
}

// 工具函数3：生成最终保存路径（当前文件目录/truck/子目录/文件名）
std::string get_save_path(const std::string& sub_dir, const std::string& filename) {
    // 1. 获取当前 .cpp 文件所在目录（依赖 get_current_file_dir() 返回正确的绝对路径）
    std::string current_file_dir = get_current_file_dir();
    fs::path current_path = current_file_dir;  // 自动兼容 Windows/Linux 路径格式

    // 2. 向上回溯两级目录，得到 /ssd/open_source/altro-cpp/（或 Windows 下的 C:\ssd\open_source\altro-cpp\）
    // 回溯逻辑：examples/ → test/ → altro-cpp/（与系统无关，按目录层级回溯）
    fs::path root_parent = current_path.parent_path().parent_path();

    // 3. 拼接目标根目录：altro-cpp/ + Testing/Temporary/（自动使用系统分隔符）
    fs::path target_root = root_parent / "Testing" / "Temporary";

    // 4. 拼接最终目标目录：Temporary/ + truck/ + 子目录（自动处理 sub_dir 中的多级路径）
    fs::path target_dir = target_root / "truck" / fs::path(sub_dir);  // fs::path 自动解析 sub_dir 中的分隔符

    // 5. 递归创建目标目录（跨平台支持，自动创建所有多级子目录）
    // exists: 检查目录是否存在；create_directories: 递归创建（已存在则不报错）
    if (!fs::exists(target_dir)) {
        fs::create_directories(target_dir);
    }

    // 6. 拼接最终文件路径（自动添加系统兼容的分隔符）
    fs::path final_path = target_dir / filename;

    // 转换为字符串返回（兼容原有函数调用）
    return final_path.string();
}

namespace plt = matplotlibcpp;

class TruckExampleTest : public altro::problems::TruckProblem, public ::testing::Test {
 public:
  void drawPlot1(
      altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls>& solver_al, std::string id = "0");
  void drawPlot2(
      altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls>& solver_al, std::string id = "0");
  void drawPlot3(
      altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls>& solver_al, std::string id = "0");
  void drawPlot4(std::vector<double> trajx, std::vector<double> trajy);
  void drawPlot5(std::vector<double> trajx, std::vector<double> trajy);
  void drawPlot6(
      altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls>& solver_al, std::string id = "0");
  void drawPlot7(std::vector<double> trajx, std::vector<double> trajy);
  void drawPlot9(std::vector<double> trajx, std::vector<double> trajy);
//   void drawPlot8(
//       altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls>& solver_al, std::string id = "0");
//   void drawPlot9(std::vector<double> trajx, std::vector<double> trajy);

// 这里修改仿真的场景
 protected:
  void SetUp() override { SetScenario(StraightRoadWithObs); }  // LaneChange StraightRoadWithObs ForwardObs Uturn Largecurve
};

// straight road with obs
  void TruckExampleTest::drawPlot1(altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> &solver_al, std::string id) {
    // Straight Road With Obstacle
    std::vector<double> xf = {100, 0};
    // double obsr = 6;
    std::vector<double> obsx = {40, 80};
    std::vector<double> obsy = {2, -2};

    std::vector<std::vector<double>>
        xref(2, std::vector<double>(51, 0));
    std::vector<double> trajx(N+1), trajy(N+1);
    for (int i = 0; i <= N; ++i) {
      xref[0][i] = xf[0] / N * i;
      trajx[i] = solver_al.GetiLQRSolver().GetTrajectory()->State(i)[0];
      trajy[i] = solver_al.GetiLQRSolver().GetTrajectory()->State(i)[1];
    }

    plt::figure_size(1000, 230);
    plt::xlim(0, 100);
    plt::ylim(-6, 6);
    plt::plot(xref[0], xref[1], "b--");
    plt::plot({obsx[0]}, {obsy[0]}, {{"c","red"}, {"marker","o"}, {"markersize","55"}});
    plt::plot({obsx[1]}, {obsy[1]}, {{"c","red"}, {"marker","o"}, {"markersize","55"}});
    plt::plot(trajx, trajy, "g-");
    // plt::axis("scaled");
    plt::save(get_save_path("obs/variable/2obs", id + ".png"));
    // plt::show();
  }

// lane change
  void TruckExampleTest::drawPlot2(altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> &solver_al, std::string id) {
    // Straight Road With Obstacle
    std::vector<double> xf = {130, 3.75};
    // double obsr = 6;
    double t = std::atoi(id.c_str()) * 0.1;
    std::vector<double> obsx = {0, 10 + 10 * t};
    std::vector<double> obsy = {3.75, 3.75 / 2};

    std::vector<std::vector<double>> xref(2, std::vector<double>(2, 3.75)),
        xorigin(2, std::vector<double>(2, 0)), line1(2, std::vector<double>(2, 3.75 / 2.0)),
        line2(2, std::vector<double>(2, -3.75 / 2.0)), line3(2, std::vector<double>(2, 3.75 *1.5));
    xref[0][0] = 0;
    xref[0][1] = 200;
    xorigin[0][1] = 200;
    line1[0][0] = 0;
    line1[0][1] = 200;
    line2[0][0] = 0;
    line2[0][1] = 200;
    line3[0][0] = 0;
    line3[0][1] = 200;
    std::vector<double> trajx(N + 1), trajy(N + 1);
    for (int i = 0; i <= N; ++i) {
      trajx[i] = solver_al.GetiLQRSolver().GetTrajectory()->State(i)[0];
      trajy[i] = solver_al.GetiLQRSolver().GetTrajectory()->State(i)[1];
    }
    // std::cout << "u0 = " << solver_al.GetiLQRSolver().GetTrajectory()->Control(0) << std::endl;
    // std::cout << "u1 = " << solver_al.GetiLQRSolver().GetTrajectory()->Control(1) << std::endl;
    // std::cout << "u2 = " << solver_al.GetiLQRSolver().GetTrajectory()->Control(2) << std::endl;

    plt::figure_size(1400, 200);
    plt::xlim(-5, 155);
    // plt::xlim(-5, 205);
    plt::ylim(-2, 6);
    plt::plot(xref[0], xref[1], "b--");
    plt::plot(xorigin[0], xorigin[1], "b--");
    plt::plot(line1[0], line1[1], "k--");
    plt::plot(line2[0], line2[1], "k-");
    plt::plot(line3[0], line3[1], "k-");
    // plt::plot({obsx[0]}, {obsy[0]}, {{"c","red"}, {"marker","s"}, {"markersize","40"}});
    plt::plot({obsx[1]}, {obsy[1]}, {{"c","red"}, {"marker","o"}, {"markersize","38"}});
    plt::plot(trajx, trajy, "g-");
    // plt::axis("scaled");
    plt::save(get_save_path("lanechange/variable", id + ".png"));
    // plt::show();
  }

// forward obs
  void TruckExampleTest::drawPlot3(altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> &solver_al, std::string id) {
    // Straight Road With Obstacle
    std::vector<double> xf = {50, 0};
    // double obsr = 6;
    std::vector<double> obsx = {50};
    std::vector<double> obsy = {1};

    std::vector<std::vector<double>>
        xref(2, std::vector<double>(2, 0));
    xref[0][1] = 100;
    std::vector<double> trajx(N + 1), trajy(N + 1);
    for (int i = 0; i <= N; ++i) {
    //   xref[0][i] = xf[0] / N * i;
      trajx[i] = solver_al.GetiLQRSolver().GetTrajectory()->State(i)[0];
      trajy[i] = solver_al.GetiLQRSolver().GetTrajectory()->State(i)[1];
    }

    plt::figure_size(2000, 460);
    plt::xlim(-2, 102);
    plt::ylim(-6, 6);
    plt::plot(xref[0], xref[1], "b--");
    plt::plot(obsx, obsy, {{"c","red"}, {"marker","o"}, {"markersize","120"}});
    plt::plot(trajx, trajy, "g-");
    // plt::axis("scaled");
    plt::save(get_save_path("obs/variable/1obs", id + ".png"));
    // plt::show();
  }

// lane change continuously
  void TruckExampleTest::drawPlot4(std::vector<double> trajx, std::vector<double> trajy) {
    // Straight Road With Obstacle
    std::vector<double> xf = {130, 3.75};
    // double obsr = 6;
    std::vector<double> obsx = {0, 40};
    std::vector<double> obsy = {3.75, 3.75 / 2};

    std::vector<std::vector<double>> xref(2, std::vector<double>(2, 3.75)),
        xorigin(2, std::vector<double>(2, 0)), line1(2, std::vector<double>(2, 3.75 / 2.0)),
        line2(2, std::vector<double>(2, -3.75 / 2.0)), line3(2, std::vector<double>(2, 3.75 *1.5));
    xref[0][0] = 0;
    xref[0][1] = 200;
    xorigin[0][1] = 200;
    line1[0][0] = 0;
    line1[0][1] = 200;
    line2[0][0] = 0;
    line2[0][1] = 200;
    line3[0][0] = 0;
    line3[0][1] = 200;

    plt::figure_size(1400, 200);
    plt::xlim(-5, 155);
    // plt::xlim(-5, 205);
    plt::ylim(-2, 6);
    plt::plot(xref[0], xref[1], "b--");
    plt::plot(xorigin[0], xorigin[1], "b--");
    plt::plot(line1[0], line1[1], "k--");
    plt::plot(line2[0], line2[1], "k-");
    plt::plot(line3[0], line3[1], "k-");
    // plt::plot({obsx[0]}, {obsy[0]}, {{"c","red"}, {"marker","s"}, {"markersize","40"}});
    // plt::plot({obsx[1]}, {obsy[1]}, {{"c","red"}, {"marker","o"}, {"markersize","38"}});
    plt::plot(trajx, trajy, "g-");
    // plt::axis("scaled");
    plt::save(get_save_path("lanechange/variable", "lanechange.png"));
    // plt::show();
  }

// straight road with obs continuously
  void TruckExampleTest::drawPlot5(std::vector<double> trajx, std::vector<double> trajy) {
    // Straight Road With Obstacle
    std::vector<double> xf = {100, 0};
    // double obsr = 6;
    std::vector<double> obsx = {40, 80};
    std::vector<double> obsy = {2, -2};

    std::vector<std::vector<double>>
        xref(2, std::vector<double>(51, 0));
    for (int i = 0; i <= N; ++i) {
      xref[0][i] = xf[0] / N * i;
    }

    plt::figure_size(1000, 230);
    plt::xlim(0, 100);
    plt::ylim(-6, 6);
    plt::plot(xref[0], xref[1], "b--");
    plt::plot({obsx[0]}, {obsy[0]}, {{"c","red"}, {"marker","o"}, {"markersize","55"}});
    plt::plot({obsx[1]}, {obsy[1]}, {{"c","red"}, {"marker","o"}, {"markersize","55"}});
    plt::plot(trajx, trajy, "g-");
    // plt::axis("scaled");
    plt::save(get_save_path("obs/variable/2obs", "StraightRoadWithObsCon.png"));
    // plt::show();
  }

// Uturn
 void TruckExampleTest::drawPlot6(altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> &solver_al, std::string id) {

    std::vector<double> trajx(N+1), trajy(N+1);
    for (int i = 0; i <= N; ++i) {
      trajx[i] = solver_al.GetiLQRSolver().GetTrajectory()->State(i)[0];
      trajy[i] = solver_al.GetiLQRSolver().GetTrajectory()->State(i)[1];
    }
    // std::vector<std::vector<int>> xref = { {0, 40, 40, 0},
    //                                        {0, 0, 20, 20}};
    std::vector<std::vector<int>> xref = { {0, 40},
                                           {0, 0}};
    std::vector<std::vector<double>> xref1 = { {40, 40},
                                            {7.5, 20}};
    std::vector<std::vector<double>> xref2 = { {32.5, 0},
                                            {20, 20}};
    // std::vector<std::vector<double>> xref(2, std::vector<double>(63));
    // xref[0][0] = 0;
    // xref[1][0] = 0;
    // xref[0][62] = 0;
    // xref[1][62] = 20;
    // for (int i = 1; i <= 61; i++) {
    //   xref[0][i] = 35.0 + 10.0 * sin(M_PI * (i - 1) / 60.0);
    //   xref[1][i] = 10.0 - 10.0 * cos(M_PI * (i - 1) / 60.0);
    // }
    plt::figure_size(1000, 400);
    plt::xlim(-2, 42);
    plt::ylim(-2, 22);
    plt::plot(xref[0], xref[1], "b--");
    plt::plot(xref1[0], xref1[1], "b--");
    plt::plot(xref2[0], xref2[1], "b--");
    plt::plot(trajx, trajy, "g-");
    // plt::axis("scaled");
    plt::save(get_save_path("uturn", id + ".png"));
    // plt::show();
  }

// Uturn continuously
  void TruckExampleTest::drawPlot7(std::vector<double> trajx, std::vector<double> trajy) {
    // std::vector<std::vector<int>> xref = { {0, 40, 40, 0},
    //                                        {0, 0, 20, 20}};
    std::vector<std::vector<int>> xref = { {0, 40},
                                           {0, 0}};
    std::vector<std::vector<double>> xref1 = { {40, 40},
                                            {7.5, 20}};
    std::vector<std::vector<double>> xref2 = { {32.5, 0},
                                            {20, 20}};
    // std::vector<std::vector<double>> xref(2, std::vector<double>(63));
    // xref[0][0] = 0;
    // xref[1][0] = 0;
    // xref[0][62] = 0;
    // xref[1][62] = 20;
    // for (int i = 1; i <= 61; i++) {
    //   xref[0][i] = 35.0 + 10.0 * sin(M_PI * (i - 1) / 60.0);
    //   xref[1][i] = 10.0 - 10.0 * cos(M_PI * (i - 1) / 60.0);
    // }
    plt::figure_size(1000, 400);
    plt::xlim(-2, 42);
    plt::ylim(-2, 22);
    plt::plot(xref[0], xref[1], "b--");
    plt::plot(xref1[0], xref1[1], "b--");
    plt::plot(xref2[0], xref2[1], "b--");
    plt::plot(trajx, trajy, "g-");
    // plt::axis("scaled");
    plt::save(get_save_path("uturn", "uturncon.png"));
    // plt::show();
  }

// // Largecurve
//  void TruckExampleTest::drawPlot8(altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> &solver_al, std::string id) {
//     std::vector<double> trajx(N+1), trajy(N+1);
//     for (int i = 0; i <= N; ++i) {
//       trajx[i] = solver_al.GetiLQRSolver().GetTrajectory()->State(i)[0];
//       trajy[i] = solver_al.GetiLQRSolver().GetTrajectory()->State(i)[1];
//     }
//     std::vector<std::vector<double>> xref(103, std::vector<double>(2));
//     xref[0] = {0, 0};
//     xref[102] = {0, 60};
//     for (int i = 1; i <= 101; i++) {
//       xref[i][0] = 50 + 30 * sin(M_PI * (i - 1) / 100);
//       xref[i][0] = 30 - 30 * cos(M_PI * (i - 1) / 100);
//     }
//     plt::figure_size(2000, 400);
//     plt::xlim(0, 60);
//     plt::ylim(-12, 12);
//     plt::plot(xref[0], xref[1], "b--");
//     plt::plot(trajx, trajy, "g-");
//     plt::axis("scaled");
//     plt::save(get_save_path(""", id + ".png"));
//     // plt::show();
//   }
//   // Largecurve continuously
//   void TruckExampleTest::drawPlot9(std::vector<double> trajx, std::vector<double> trajy) {
//     std::vector<std::vector<double>> xref(103, std::vector<double>(2));
//     xref[0] = {0, 0};
//     xref[102] = {0, 60};
//     for (int i = 1; i <= 101; i++) {
//       xref[i][0] = 50 + 30 * sin(M_PI * (i - 1) / 100);
//       xref[i][0] = 30 - 30 * cos(M_PI * (i - 1) / 100);
//     }
//     plt::figure_size(2000, 400);
//     plt::xlim(0, 60);
//     plt::ylim(-12, 12);
//     plt::plot(xref[0], xref[1], "b--");
//     plt::plot(trajx, trajy, "g-");
//     plt::axis("scaled");
//     plt::save(get_save_path("largecurvecon",  + ".png"));
//     // plt::show();
//   }

// forward obs continue
  void TruckExampleTest::drawPlot9(std::vector<double> trajx, std::vector<double> trajy) {
    // Straight Road With Obstacle
    std::vector<double> xf = {50, 0};
    // double obsr = 6;
    std::vector<double> obsx = {50};
    std::vector<double> obsy = {1};

    std::vector<std::vector<double>>
        xref(2, std::vector<double>(2, 0));
    xref[0][1] = 100;

    plt::figure_size(1400, 200);
    plt::xlim(-2, 102);
    plt::ylim(-6, 6);
    plt::plot(xref[0], xref[1], "b--");
    plt::plot(obsx, obsy, {{"c","red"}, {"marker","o"}, {"markersize","50"}});
    plt::plot(trajx, trajy, "g-");
    // plt::axis("scaled");
    plt::save(get_save_path("obs/variable/1obs", "ForwardCon.png"));
    // plt::show();
  }

TEST_F(TruckExampleTest, SolveContinuously) {
  Eigen::Vector4d x_init(0, 0, 0, 0);
  std::vector<double> trajx(1, 0), trajy(1, 0);
  setenv("MPLBACKEND", "Agg", 1);
  // double max_steer = 0.0;

  // LaneChange StraightRoadWithObs ForwardObs Uturn Largecurve
  SetScenario(ForwardObs);
  // 获取当前场景（从 TruckProblem 继承的 GetScenario()）
  Scenario current_scenario = GetScenario();

  for (int i = 0; i < 100; i++) {
    altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver_al = MakeALSolver(x_init);
    solver_al.SetPenalty(10.0);
    solver_al.GetOptions().verbose = altro::LogLevel::kDebug;
    solver_al.Solve();
    x_init = solver_al.GetiLQRSolver().GetTrajectory()->State(1);
    trajx.push_back(x_init(0));
    trajy.push_back(x_init(1));

    // 分场景调用单步绘图函数
    switch (current_scenario) {
      case StraightRoadWithObs:
        drawPlot1(solver_al, std::to_string(i));  // 双障碍物直道场景
        break;
      case LaneChange:
        drawPlot2(solver_al, std::to_string(i));  // 车道变换场景
        break;
      case ForwardObs:
        drawPlot3(solver_al, std::to_string(i));  // 前方单障碍物场景
        break;
      case Uturn:
        drawPlot6(solver_al, std::to_string(i));  // 掉头场景
        break;
      // 如需支持 Largecurve，需先完善 drawPlot8
      // case Largecurve:
      //   drawPlot8(solver_al, std::to_string(i));
      //   break;
      default:
        GTEST_FAIL() << "Unsupported scenario for continuous solving: " << current_scenario;
    }

    // max_steer = std::max(max_steer, solver_al.GetiLQRSolver().GetTrajectory()->Control(0)(0));
  }
  // std::cout << max_steer << std::endl;
  // 分场景调用连续轨迹绘图函数
  switch (current_scenario) {
    case StraightRoadWithObs:
      drawPlot5(trajx, trajy);  // 双障碍物直道连续轨迹
      break;
    case LaneChange:
      drawPlot4(trajx, trajy);  // 车道变换连续轨迹
      break;
    case ForwardObs:
      drawPlot9(trajx, trajy);  // 前方单障碍物连续轨迹
      break;
    case Uturn:
      drawPlot7(trajx, trajy);  // 掉头连续轨迹
      break;
    // case Largecurve:
    //   drawPlot9(trajx, trajy);  // 需完善 Largecurve 连续绘图函数
    //   break;
    default:
      GTEST_FAIL() << "Unsupported scenario for continuous trajectory plot: " << current_scenario;
  }

  EXPECT_EQ(1, 1);
}

TEST_F(TruckExampleTest, SolveConstrained) {
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver_al = MakeALSolver();
  solver_al.SetPenalty(10.0);
  solver_al.GetOptions().verbose = altro::LogLevel::kDebug;
  solver_al.Solve();

  // drawPlot3(solver_al);

  int num_obstacles = cx.size();
  for (int k = 0; k <= N; ++k) {
    double px = solver_al.GetiLQRSolver().GetTrajectory()->State(k)[0];
    double py = solver_al.GetiLQRSolver().GetTrajectory()->State(k)[1];
    for (int i = 0; i < num_obstacles; ++i) {
      altro::examples::Circle obs(cx(i), cy(i), cr(i));
      EXPECT_GT(obs.Distance(px, py), -1e-3); // 1 mm
    }
  }

  EXPECT_EQ(solver_al.GetStatus(), altro::SolverStatus::kSolved);
  EXPECT_LT(solver_al.MaxViolation(), solver_al.GetOptions().constraint_tolerance);
  EXPECT_LT(solver_al.GetStats().cost_decrease.back(), solver_al.GetOptions().cost_tolerance);
  EXPECT_LT(solver_al.GetStats().gradient.back(), solver_al.GetOptions().gradient_tolerance);
}

// TEST_F(TruckExampleTest, Construction) {
//   altro::ilqr::iLQR<NStates, NControls> solver = MakeSolver();
//   double J = solver.Cost();
//   const double J_expected = 133.1151550141444;  // from Altro.jl
//   EXPECT_LT(std::abs(J - J_expected), 1e-6);
//   const bool al_cost = true;
//   altro::ilqr::iLQR<NStates, NControls> solver_al = MakeSolver(al_cost);
//   J = solver_al.Cost();
//   const double Jal_expected = 141.9639680271223;
//   EXPECT_LT(std::abs(J - Jal_expected), 1e-6);
// }

// TEST_F(TruckExampleTest, IncreasePenalty) {
//   altro::problem::Problem prob = MakeProblem(true);

//   // Create iLQR solver w/ AL objective
//   const bool al_cost = true;
//   altro::ilqr::iLQR<NStates, NControls> solver = MakeSolver(al_cost);
//   double J0 = solver.Cost();

//   // Create AL-iLQR Solver
//   altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver_al(prob);
//   solver_al.SetTrajectory(std::make_shared<altro::Trajectory<NStates, NControls>>(InitialTrajectory()));
//   solver_al.GetiLQRSolver().Rollout();
//   double J = solver_al.GetiLQRSolver().Cost();

//   EXPECT_DOUBLE_EQ(J0, J);

//   solver_al.SetPenalty(10.0);
//   double J_expected = 221.6032851439234;  // from Altro.jl
//   J = solver_al.GetiLQRSolver().Cost();
//   EXPECT_LT(std::abs(J_expected - J), 1e-6);
// }

// TEST_F(TruckExampleTest, SolveOneStep) {
//   altro::problem::Problem prob = MakeProblem(true);
//   altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver_al(prob);
//   solver_al.SetTrajectory(std::make_shared<altro::Trajectory<NStates, NControls>>(InitialTrajectory()));
//   solver_al.GetiLQRSolver().Rollout();
//   solver_al.SetPenalty(10.0);

//   altro::ilqr::iLQR<NStates, NControls>& ilqr = solver_al.GetiLQRSolver();
//   ilqr.Solve();

//   solver_al.UpdateDuals();
//   solver_al.UpdatePenalties();

//   Eigen::VectorXd lambdaN(n);
//   lambdaN << 0.43555910438329626, -0.5998598475208317, 0.0044282251970790935, 1; // from Altro.jl
//   EXPECT_TRUE(solver_al.GetALCost(N)->GetEqualityConstraints()[0]->GetDuals().isApprox(-lambdaN, 1e-6));
// }


// TEST_F(TruckExampleTest, SolveParallel) {
//   N = 40;
//   altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver_al = MakeALSolver();
//   // solver_al.SetPenalty(10.0);
//   solver_al.GetOptions().initial_penalty = 10.0;
//   solver_al.GetOptions().verbose = altro::LogLevel::kDebug;

//   altro::Trajectory<NStates, NControls> Z0 = InitialTrajectory<NStates, NControls>();
//   solver_al.Init();


//   // Compare expansions after 2 iLQR solves
//   altro::ilqr::iLQR<NStates, NControls>& ilqr = solver_al.GetiLQRSolver();
//   *(ilqr.GetTrajectory()) = Z0;
//   auto pen = solver_al.GetALCost(0)->GetInequalityConstraints()[0]->GetPenalty();
//   EXPECT_TRUE(pen.isApproxToConstant(10.0));
//   ilqr.SolveSetup();
//   ilqr.Rollout();

//   auto step = [&ilqr](int iter) {
//     for (int i = 0; i < iter; ++i) {
//       ilqr.UpdateExpansions();
//       ilqr.BackwardPass();
//       ilqr.ForwardPass();
//       ilqr.UpdateConvergenceStatistics();
//     }
//   };

//   step(2);
//   std::vector<Eigen::MatrixXd> jacs;
//   std::vector<Eigen::MatrixXd> Qxx;
//   std::vector<Eigen::MatrixXd> Quu;
//   std::vector<Eigen::MatrixXd> K;
//   for (int k = 0; k < ilqr.NumSegments(); ++k) {
//     jacs.emplace_back(ilqr.GetKnotPointFunction(k).GetDynamicsExpansion().GetJacobian());
//     Qxx.emplace_back(ilqr.GetKnotPointFunction(k).GetCostExpansion().dxdx());
//     Quu.emplace_back(ilqr.GetKnotPointFunction(k).GetCostExpansion().dudu());
//     K.emplace_back(ilqr.GetKnotPointFunction(k).GetFeedbackGain());
//   }

//   *(ilqr.GetTrajectory()) = Z0;
//   ilqr.GetStats().Reset();
//   ilqr.GetOptions().nthreads = 2;

//   ilqr.SolveSetup();
//   ilqr.Rollout();
//   step(2);
//   for (int k = 0; k < ilqr.NumSegments(); ++k) {
//     Eigen::MatrixXd jac = ilqr.GetKnotPointFunction(k).GetDynamicsExpansion().GetJacobian();
//     bool dynamics = jacs[k].isApprox(jac);
//     bool qxx = Qxx[k].isApprox(ilqr.GetKnotPointFunction(k).GetCostExpansion().dxdx());
//     bool quu = Quu[k].isApprox(ilqr.GetKnotPointFunction(k).GetCostExpansion().dudu());
//     bool gain = K[k].isApprox(ilqr.GetKnotPointFunction(k).GetFeedbackGain());
//     // fmt::print("Index {}: Dynamics? {}, Qxx? {}, Quu? {}, Gain? {}\n", k, dynamics, qxx, quu, gain);
//     // if (!dynamics) {
//     //   fmt::print("Expected:\n{}\n", jacs[k]);
//     //   fmt::print("Got:\n{}\n", jac);
//     // }
//     EXPECT_TRUE(dynamics);
//     EXPECT_TRUE(qxx);
//     EXPECT_TRUE(quu);
//     EXPECT_TRUE(gain);
//   }

//   // Compare entire solves
//   *(ilqr.GetTrajectory()) = Z0;
//   solver_al.GetOptions().nthreads = 1;
//   solver_al.Solve();
//   double cost = ilqr.Cost();
//   int iters = solver_al.GetStats().iterations_total;

//   *(ilqr.GetTrajectory()) = Z0;
//   solver_al.GetOptions().nthreads = 2;
//   solver_al.Solve();
//   EXPECT_DOUBLE_EQ(cost, ilqr.Cost());
//   EXPECT_EQ(iters, ilqr.GetStats().iterations_total);
// }