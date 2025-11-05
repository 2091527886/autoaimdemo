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
find_package(TensorRT-YOLO REQUIRED)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8)
endif()

# 2) 可执行目标
add_executable(aimassistant
  main.cpp
  OverlayAnimator.cpp
  Mouse.cpp
  PidController.cpp
  SystemTuner.cpp
  ObjDet.cpp
  ScreenCapturer.cpp
  CaptureDXGI.cpp
  CaptureWGC.cpp
  CapAsync.cpp
  DetAsync.cpp
  ByteTrack.cpp
  TrkAsync.cpp
  Game_overlay.cpp
  OvlAsync.cpp
  LatCsvAsync.cpp
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

