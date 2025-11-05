# cd altro-cpp         # Change directory into directory root.
mkdir build -p         # Create a build directory.
cd build
cmake ..               # Run the CMake configuration step. 
cmake --build .        # Build all CMake targets
make -j$(( $(nproc) / 2 ))
# ctest .