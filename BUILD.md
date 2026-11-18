# AimAssistant 编译方法

## 环境要求

- Windows x64
- Visual Studio（包含 C++ 桌面开发工具；本次验证使用 Visual Studio 18 Community）
- CMake 3.20 或更高版本（本次验证：4.4.4）
- CUDA Toolkit（本次验证：13.5）
- OpenCV 开发包及 `OpenCVConfig.cmake`
- TensorRT 10.9.0.34

项目还会编译 `TensorRT-YOLO` 子目录，因此 TensorRT 的头文件和 `.lib` 文件必须可用。

## 本机已验证的编译命令

在项目根目录执行 PowerShell：

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat\" -arch=amd64 && cmake -S . -B out/build/nmake-release -G \"NMake Makefiles\" -DCMAKE_BUILD_TYPE=Release -DTRT_PATH=\"C:/cuda/TensorRT-10.9.0.34\" -DOpenCV_DIR=\"C:/Users/sb/Desktop/autoaimcmake/opencv/build/x64/vc16/lib\""
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat\" -arch=amd64 && cmake --build out/build/nmake-release --parallel 4"
```

如果 Visual Studio、TensorRT 或 OpenCV 安装路径不同，请替换命令中的路径：

- `VsDevCmd.bat`：Visual Studio 开发者命令脚本
- `TRT_PATH`：TensorRT 安装目录
- `OpenCV_DIR`：包含 `OpenCVConfig.cmake` 的目录

也可以合并为一次执行：

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat\" -arch=amd64 && cmake -S . -B out/build/nmake-release -G \"NMake Makefiles\" -DCMAKE_BUILD_TYPE=Release -DTRT_PATH=\"C:/cuda/TensorRT-10.9.0.34\" -DOpenCV_DIR=\"C:/Users/sb/Desktop/autoaimcmake/opencv/build/x64/vc16/lib\" && cmake --build out/build/nmake-release --parallel 4"
```

## 编译产物

成功后主要产物位于 `out/build/nmake-release`：

- `aimassistant.exe`
- `trtyolo.dll`
- `IbInputSimulator.dll`
- `TensorRT-YOLO/modules/plugin/custom_plugins.dll`

## 关于 CMake 预设

`CMakePresets.json` 中的 `win-vs-release` 使用 Ninja。如果本机已安装 Ninja，可以执行：

```powershell
cmake --preset win-vs-release
cmake --build --preset win-vs-release
```

本次环境没有安装 Ninja，因此使用 Visual Studio 开发者环境中的 `NMake Makefiles` 完成了等价的 Release 编译。

## 常见问题

### 找不到 OpenCV

将 `OpenCV_DIR` 设置为实际包含 `OpenCVConfig.cmake` 的目录，例如：

```text
-DOpenCV_DIR=C:/path/to/opencv/build/x64/vc16/lib
```

### 找不到 Ninja

安装 Ninja，或按本文使用 `NMake Makefiles`。使用 NMake 时必须先调用 `VsDevCmd.bat`，使 `cl.exe`、`link.exe` 和 `nmake.exe` 进入 PATH。

### CUDA 13.5 / 新版 MSVC 预处理器错误

项目的 `CMakeLists.txt` 已包含以下兼容设置：

- `_SILENCE_EXPERIMENTAL_COROUTINE_DEPRECATION_WARNINGS`
- `/Zc:preprocessor`
- CUDA host 编译器的 `-Xcompiler=/Zc:preprocessor`

这些设置用于兼容 WinRT 头文件和 CUDA CCCL；修改工具链版本后应重新配置构建目录。
