cmake_minimum_required(VERSION 3.20)

project(AimAssistant LANGUAGES CXX)

# ---- Rovo Dev: per-target 高优化与指令集设置 ----
include(CheckCXXCompilerFlag)

function(rovo_check_flag result_flag flag)
  unset(${result_flag} CACHE)
  check_cxx_compiler_flag("${flag}" ${result_flag})
endfunction()

function(rovo_enable_optimizations tgt)
  if(NOT TARGET ${tgt})
    message(FATAL_ERROR "Target ${tgt} 不存在，先 add_executable/add_library 再调用 rovo_enable_optimizations()")
  endif()

  # 仅在 Release 构建下添加
  set(cfg "$<CONFIG:Release>")

  if(MSVC)
    target_compile_options(${tgt} PRIVATE
      "$<${cfg}:/O2>"
      "$<${cfg}:/GL>"       # 编译期 LTCG
    )
    target_link_options(${tgt} PRIVATE
      "$<${cfg}:/LTCG>"
    )

    # 指令集：优先 AVX512，否则 AVX2
    rovo_check_flag(HAS_ARCH_AVX512 "/arch:AVX512")
    if(HAS_ARCH_AVX512)
      target_compile_options(${tgt} PRIVATE "$<${cfg}:/arch:AVX512>")
    else()
      rovo_check_flag(HAS_ARCH_AVX2 "/arch:AVX2")
      if(HAS_ARCH_AVX2)
        target_compile_options(${tgt} PRIVATE "$<${cfg}:/arch:AVX2>")
      endif()
    endif()

  else() # GCC/Clang
    target_compile_options(${tgt} PRIVATE
      "$<${cfg}:-O3>"
      "$<${cfg}:-DNDEBUG>"
      "$<${cfg}:-flto>"
    )
    target_link_options(${tgt} PRIVATE "$<${cfg}:-flto>")

    rovo_check_flag(HAS_MAVX512F "-mavx512f")
    if(HAS_MAVX512F)
      target_compile_options(${tgt} PRIVATE "$<${cfg}:-mavx512f>")
      target_link_options(${tgt} PRIVATE "$<${cfg}:-mavx512f>")
    else()
      rovo_check_flag(HAS_MAVX2 "-mavx2")
      if(HAS_MAVX2)
        target_compile_options(${tgt} PRIVATE "$<${cfg}:-mavx2>")
        target_link_options(${tgt} PRIVATE "$<${cfg}:-mavx2>")
      endif()
    endif()
  endif()
endfunction()
# ---- Rovo Dev: end ----

# 1) 基础编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(OpenCV REQUIRED)
find_package(TensorRT-YOLO REQUIRED)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8)
  # 可选：静态运行时（如需要）
  # string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  # string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

# 2) 可执行目标
add_executable(aimassistant
  main.cpp
)

# 启用 Release 高优化/LTO/AVX
rovo_enable_optimizations(aimassistant)

# 3) 头文件与宏定义
# 根目录包含，以便找到 InputSimulator.hpp
# 如果有其它头文件目录，可追加到此列表
set(PROJECT_ROOT_INCLUDE "${CMAKE_SOURCE_DIR}")

target_include_directories(aimassistant PRIVATE
  ${PROJECT_ROOT_INCLUDE}
)

# 常见 Windows 定义，防止宏冲突并启用 Unicode
if(WIN32)
  target_compile_definitions(aimassistant PRIVATE
    NOMINMAX
    UNICODE
    _UNICODE
    WIN32_LEAN_AND_MEAN
  )
endif()

# 4) 平台/系统库链接（Windows + DirectX）
if(WIN32)
  target_link_libraries(aimassistant PRIVATE d3d11 dxgi)
endif()

# 5) OpenCV：直接在 CMakeLists.txt 中硬编码路径（不读取 depend.txt，不使用 find_package）
# 请将下列三个变量设置为您机器上的实际路径/库名
# 例如：
#   set(OPENCV_INCLUDE_DIR "D:/SDKs/opencv-4.11.0/build/include")
#   set(OPENCV_LIB_DIR     "D:/SDKs/opencv-4.11.0/build/x64/vc17/lib")
#   set(OPENCV_LIBS        opencv_world4110d.lib opencv_world4110.lib)

# 应用到目标




# 6) TensorRT-YOLO：使用 find_package(CONFIG)
# 提示：请提前设置 CMAKE_PREFIX_PATH 或 <pkg>_DIR 以便找到 tensorrt-yolo-config.cmake
# 例如：
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="C:/cuda/tensorrtyolo"
# 或：
#   cmake -S . -B build -Dtensorrt-yolo_DIR="C:/cuda/tensorrtyolo"


target_include_directories(aimassistant PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT-YOLO_INCLUDE_DIRS}
)

# 私有链接库
target_link_libraries(aimassistant PRIVATE
    ${OpenCV_LIBS}
    ${TensorRT-YOLO_LIBs}
)


# 7) 复制根目录下 DLL 至运行目录（确保运行时能找到 InputSimulator DLL）
set(INPUT_SIM_DLL "${CMAKE_SOURCE_DIR}/IbInputSimulator.dll")
add_custom_command(TARGET aimassistant POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Post-build: copy IbInputSimulator.dll if present"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${INPUT_SIM_DLL}" "$<TARGET_FILE_DIR:aimassistant>"
)

# 8) 输出目录（可选）
# set_target_properties(aimassistant PROPERTIES
#   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
# )

# 9) 其他：可根据 depend.txt 进一步完善依赖
# TODO: 读取/合并 depend.txt 的内容到此 CMake；待用户提供文件内容后完善。
