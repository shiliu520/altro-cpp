# cd altro-cpp                         # Change directory into directory root.
mkdir build -p                         # Create a build directory.
cd build                               # Change directory into the build directory.
cmake ..                               # Run the CMake configuration step.

# for gdb debugging
# cmake -D ALTRO_BUILD_DOCS=ON -DCMAKE_BUILD_TYPE=Debug ..        # Build all CMake targets
# cmake --build . --target altro_docs
# make -j4

# for release build
cmake -D ALTRO_BUILD_DOCS=ON ..        # Build all CMake targets
cmake --build . --target altro_docs
make -j6

# ctest .