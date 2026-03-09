# Windows ARM64 with Clang (clang-cl).
# Required so ggml's ARM NEON code is built (it is disabled for MSVC cl.exe).
# Requires: LLVM/Clang with clang-cl on PATH (e.g. "C++ Clang compiler for Windows" from Visual Studio Installer).

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR ARM64)

set(TARGET_ARCH aarch64-pc-windows-msvc)

# Prefer clang-cl from PATH (e.g. from VS or standalone LLVM)
if(DEFINED ENV{LLVM_INSTALL_DIR})
  set(CLANG_CL "$ENV{LLVM_INSTALL_DIR}/bin/clang-cl.exe")
else()
  set(CLANG_CL "clang-cl")
endif()

set(CMAKE_C_COMPILER "${CLANG_CL}")
set(CMAKE_CXX_COMPILER "${CLANG_CL}")
set(CMAKE_C_COMPILER_TARGET ${TARGET_ARCH})
set(CMAKE_CXX_COMPILER_TARGET ${TARGET_ARCH})

set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} --target=${TARGET_ARCH}")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} --target=${TARGET_ARCH} /EHsc")
# Force C++ exceptions so gguf.cpp and other C++ code using try/catch build (clang-cl can have them disabled)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc" CACHE STRING "C++ compiler flags" FORCE)
