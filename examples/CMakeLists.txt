add_library(cost_function_examples
  quadratic_cost.cpp
  quadratic_cost.hpp
  traj_plan_cost.cpp
  traj_plan_cost.hpp
  reference_line.cpp
  reference_line.hpp
)
target_link_libraries(cost_function_examples
  PUBLIC
  problem

  PRIVATE
  utils
)

add_library(dynamics_examples
  unicycle.hpp
  unicycle.cpp
  triple_integrator.cpp
  triple_integrator.hpp
  truck.hpp
  truck.cpp
  car_kin.hpp
  car_kin.cpp
)
target_link_libraries(dynamics_examples
  PUBLIC
  problem

  PRIVATE
  utils
)

add_library(constraint_examples
  basic_constraints.cpp
  basic_constraints.hpp
  obstacle_constraints.cpp
  obstacle_constraints.hpp
)
target_link_libraries(constraint_examples
  PUBLIC
  constraints

  PRIVATE
  utils
)

add_subdirectory(problems)

# Example Executables
add_executable(unicycle
  unicycle_main.cpp
  )
target_link_libraries(unicycle
  PRIVATE
  altro
  constraint_examples
  cost_function_examples
  dynamics_examples
  example_problems
)

add_subdirectory(cartpole)