# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"

# The command to remove a file.
RM = "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:/Users/dean/OneDrive/Workspace/Projects/Seayon

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build

# Include any dependencies generated for this target.
include Tests/CMakeFiles/fit.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Tests/CMakeFiles/fit.dir/compiler_depend.make

# Include the progress variables for this target.
include Tests/CMakeFiles/fit.dir/progress.make

# Include the compile flags for this target's objects.
include Tests/CMakeFiles/fit.dir/flags.make

Tests/CMakeFiles/fit.dir/src/fit.cpp.obj: Tests/CMakeFiles/fit.dir/flags.make
Tests/CMakeFiles/fit.dir/src/fit.cpp.obj: Tests/CMakeFiles/fit.dir/includes_CXX.rsp
Tests/CMakeFiles/fit.dir/src/fit.cpp.obj: C:/Users/dean/OneDrive/Workspace/Projects/Seayon/Tests/src/fit.cpp
Tests/CMakeFiles/fit.dir/src/fit.cpp.obj: Tests/CMakeFiles/fit.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Tests/CMakeFiles/fit.dir/src/fit.cpp.obj"
	cd C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build/Tests && C:/msys64/mingw64/bin/g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Tests/CMakeFiles/fit.dir/src/fit.cpp.obj -MF CMakeFiles/fit.dir/src/fit.cpp.obj.d -o CMakeFiles/fit.dir/src/fit.cpp.obj -c C:/Users/dean/OneDrive/Workspace/Projects/Seayon/Tests/src/fit.cpp

Tests/CMakeFiles/fit.dir/src/fit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fit.dir/src/fit.cpp.i"
	cd C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build/Tests && C:/msys64/mingw64/bin/g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/dean/OneDrive/Workspace/Projects/Seayon/Tests/src/fit.cpp > CMakeFiles/fit.dir/src/fit.cpp.i

Tests/CMakeFiles/fit.dir/src/fit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fit.dir/src/fit.cpp.s"
	cd C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build/Tests && C:/msys64/mingw64/bin/g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/dean/OneDrive/Workspace/Projects/Seayon/Tests/src/fit.cpp -o CMakeFiles/fit.dir/src/fit.cpp.s

# Object files for target fit
fit_OBJECTS = \
"CMakeFiles/fit.dir/src/fit.cpp.obj"

# External object files for target fit
fit_EXTERNAL_OBJECTS =

bin/Release/Tests/fit.exe: Tests/CMakeFiles/fit.dir/src/fit.cpp.obj
bin/Release/Tests/fit.exe: Tests/CMakeFiles/fit.dir/build.make
bin/Release/Tests/fit.exe: Tests/CMakeFiles/fit.dir/linklibs.rsp
bin/Release/Tests/fit.exe: Tests/CMakeFiles/fit.dir/objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/Release/Tests/fit.exe"
	cd C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build/Tests && "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" -E rm -f CMakeFiles/fit.dir/objects.a
	cd C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build/Tests && C:/msys64/mingw64/bin/ar.exe qc CMakeFiles/fit.dir/objects.a @CMakeFiles/fit.dir/objects1.rsp
	cd C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build/Tests && C:/msys64/mingw64/bin/g++.exe -O3 -Wl,--whole-archive CMakeFiles/fit.dir/objects.a -Wl,--no-whole-archive -o ../bin/Release/Tests/fit.exe -Wl,--out-implib,libfit.dll.a -Wl,--major-image-version,0,--minor-image-version,0 @CMakeFiles/fit.dir/linklibs.rsp

# Rule to build all files generated by this target.
Tests/CMakeFiles/fit.dir/build: bin/Release/Tests/fit.exe
.PHONY : Tests/CMakeFiles/fit.dir/build

Tests/CMakeFiles/fit.dir/clean:
	cd C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build/Tests && $(CMAKE_COMMAND) -P CMakeFiles/fit.dir/cmake_clean.cmake
.PHONY : Tests/CMakeFiles/fit.dir/clean

Tests/CMakeFiles/fit.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" C:/Users/dean/OneDrive/Workspace/Projects/Seayon C:/Users/dean/OneDrive/Workspace/Projects/Seayon/Tests C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build/Tests C:/Users/dean/OneDrive/Workspace/Projects/Seayon/build/Tests/CMakeFiles/fit.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Tests/CMakeFiles/fit.dir/depend

