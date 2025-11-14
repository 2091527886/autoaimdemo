cmake_minimum_required(VERSION 3.20)

project(AimAssistant LANGUAGES CXX)


include(CheckCXXCompilerFlag)

function(check_flag result_flag flag)
  unset(${result_flag} CACHE)
  check_cxx_compiler_flag("${flag}" ${result_flag})
endfunction()

function(enable_optimizations tgt)
  if(NOT TARGET ${tgt})
    message(FATAL_ERROR "Target ${tgt} 不存在，先 add_executable/add_library 再调用 enable_optimizations()")
  endif()

  # 仅在 Release 构建下添加
  set(cfg "$<CONFIG:Release>")
    target_compile_options(${tgt} PRIVATE
      "$<${cfg}:/O2>"
      "$<${cfg}:/GL>"       # 编译期 LTCG
    )
    target_link_options(${tgt} PRIVATE
      "$<${cfg}:/LTCG>"
    )

    # 指令集：优先 AVX512，否则 AVX2
    check_flag(HAS_ARCH_AVX512 "/arch:AVX512")
    if(HAS_ARCH_AVX512)
      target_compile_options(${tgt} PRIVATE "$<${cfg}:/arch:AVX512>")
    else()
      check_flag(HAS_ARCH_AVX2 "/arch:AVX2")
      if(HAS_ARCH_AVX2)
        target_compile_options(${tgt} PRIVATE "$<${cfg}:/arch:AVX2>")
      endif()
    endif()
endfunction()

# 1) 基础编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(OpenCV REQUIRED)

# TensorRT-YOLO as submodule build
# Force CUDA to target only RTX 3060 (SM 86). We don't care other GPUs.
set(CMAKE_CUDA_ARCHITECTURES 86)
# Provide TensorRT install path (Windows example default)
set(TRT_PATH "C:/cuda/TensorRT-10.9.0.34" CACHE PATH "Path to TensorRT installation (e.g., C:/cuda/TensorRT-10.9.0.34)")
# Build embedded TensorRT-YOLO library and plugins
add_subdirectory(${CMAKE_SOURCE_DIR}/TensorRT-YOLO)

# Ensure MSVC compiles trtyolo with UTF-8 and NOMINMAX to avoid header parsing issues
if(MSVC)
  if(TARGET trtyolo)
    target_compile_options(trtyolo PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    target_compile_definitions(trtyolo PRIVATE NOMINMAX)
  endif()
endif()

if(MSVC)
  # Apply MSVC warnings/standards only to C++ sources, avoid leaking raw MSVC flags into NVCC
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
  )
endif()

# Copy TensorRT runtime DLLs after build (Windows only)
if(WIN32)
  set(TRT_BIN_DIR "${TRT_PATH}/lib")
  set(TRT_DLLS
    nvinfer_10.dll
    nvinfer_plugin_10.dll
    nvonnxparser_10.dll
  )
endif()

# 2) 可执行目标
add_executable(aimassistant
  pch.hpp
  main.cpp
  CtrlHandler.cpp
  Aim_init.cpp
  Aim_loop.cpp
  Aim_cap.cpp
  Aim_time.cpp
  Aim_anchor.cpp
  Aim_pred_err.cpp
  Aim_move.cpp
  Aim_ovl_dets.cpp
  Aim_ovl_cross.cpp
  Aim_ovl_lock.cpp
  Aim_log.cpp
  FpsCtl.cpp
  target_pick.cpp
  Mouse.cpp
  HotkeyManager.cpp
  SystemTuner.cpp
  ObjDet_init.cpp
  ObjDet_detect.cpp
  ObjDet_perf.cpp
  ScreenCapturer.Core.cpp
  ScreenCapturer.Dispatch.cpp
  ScreenCapturer.CaptureDXGI.cpp
  CaptureDXGI.cpp
  CaptureWGC.cpp
  Game_overlay.cpp
  ovl_window.cpp
  ovl_render.cpp
  ovl_gpu.cpp
  ovl_util.cpp
  AimAlgo.hpp
)

# 启用 Release 高优化/LTO/AVX
enable_optimizations(aimassistant)

# 3) 头文件与宏定义

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
  target_link_libraries(aimassistant PRIVATE d3d11 dxgi d2d1 dwrite windowscodecs dcomp dwmapi avrt windowsapp)
endif()

target_include_directories(aimassistant PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/TensorRT-YOLO/modules/trtyolo/infer
)

# 私有链接库
target_link_libraries(aimassistant PRIVATE
    ${OpenCV_LIBS}
    trtyolo
)


# 7) 复制运行所需 DLL 至运行目录
set(INPUT_SIM_DLL "${CMAKE_SOURCE_DIR}/IbInputSimulator.dll")
add_custom_command(TARGET aimassistant POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Post-build: copy IbInputSimulator.dll if present"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${INPUT_SIM_DLL}" "$<TARGET_FILE_DIR:aimassistant>"
)

# Post-build copy TensorRT runtime DLLs
if(WIN32)
  foreach(dll ${TRT_DLLS})
    add_custom_command(TARGET aimassistant POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo "Post-build: copy ${dll} if present"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TRT_BIN_DIR}/${dll}" "$<TARGET_FILE_DIR:aimassistant>"
    )
  endforeach()
  # Also copy built trtyolo.dll beside the executable
  add_custom_command(TARGET aimassistant POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Post-build: copy trtyolo.dll if present"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:trtyolo>" "$<TARGET_FILE_DIR:aimassistant>"
  )
endif()

