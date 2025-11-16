# cd altro-cpp                         # Change directory into directory root.
mkdir build -p                         # Create a build directory.
cd build                               # Change directory into the build directory.
cmake ..                               # Run the CMake configuration step.
cmake -D ALTRO_BUILD_DOCS=ON ..        # Build all CMake targets
cmake --build . --target altro_docs
# make -j$(( $(nproc) / 2 ))
make -j6
# ctest .