#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import ctypes
from typing import List, Optional

WIN_DEFAULT_TRT_LIB_DIR = r"C:\cuda\TensorRT-10.9.0.34\lib"

def add_windows_dll_dir(path: str):
    try:
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(path)
            print(f"[Info] Added DLL search dir: {path}")
    except Exception as e:
        print(f"[Warn] add_dll_directory failed for {path}: {e}")


def try_load_libs(candidates: List[str]) -> bool:
    loaded_any = False
    for lib in candidates:
        try:
            ctypes.CDLL(lib)
            print(f"[Info] Loaded plugin lib: {lib}")
            loaded_any = True
        except OSError as e:
            print(f"[Debug] Load fail: {lib} -> {e}")
    return loaded_any


def load_trt_plugins(logger, extra_plugins: Optional[List[str]] = None, trt_lib_dir: Optional[str] = None):
    import tensorrt as trt

    # 默认 Windows lib 目录
    if os.name == "nt" and not trt_lib_dir and os.path.isdir(WIN_DEFAULT_TRT_LIB_DIR):
        trt_lib_dir = WIN_DEFAULT_TRT_LIB_DIR
        print(f"[Info] Using default --trt-lib-dir: {trt_lib_dir}")

    # 将 TRT lib 目录加入 DLL 搜索路径（Windows）
    if trt_lib_dir and os.path.isdir(trt_lib_dir):
        add_windows_dll_dir(trt_lib_dir)
    elif trt_lib_dir:
        print(f"[Warn] --trt-lib-dir 不存在或不可访问: {trt_lib_dir}")

    # 初始化插件注册表
    try:
        trt.init_libnvinfer_plugins(logger, "")
        print("[Info] init_libnvinfer_plugins OK")
    except Exception as e:
        print("[Warn] init_libnvinfer_plugins failed:", e)

    # 仅尝试无后缀与 10.x 后缀 DLL
    if os.name == "nt":  # Windows
        candidates = [
            "nvinfer_plugin.dll",
            "nvinfer.dll",
            "nvinfer_plugin_10.dll",
            "nvinfer_10.dll",
        ]
    else:
        candidates = [
            "libnvinfer_plugin.so",
            "libnvinfer.so",
            "libnvinfer_plugin.so.10",
            "libnvinfer.so.10",
        ]

    any_loaded = try_load_libs(candidates)
    if not any_loaded:
        print("[Warn] Could not auto-load TensorRT core/plugin libs from system path. Consider using --trt-lib-dir or --plugins.")

    # 用户指定的额外插件库
    for p in (extra_plugins or []):
        try:
            ctypes.CDLL(p)
            print(f"[Info] Loaded user plugin lib: {p}")
        except OSError as e:
            print(f"[Warn] Failed to load user plugin lib: {p} -> {e}")


def dtype_to_str(trt, dt):
    try:
        if dt == trt.DataType.FLOAT:
            return "FLOAT"
        if dt == trt.DataType.HALF:
            return "HALF"
        if dt == trt.DataType.INT8:
            return "INT8"
        if dt == trt.DataType.INT32:
            return "INT32"
        if hasattr(trt.DataType, "BOOL") and dt == trt.DataType.BOOL:
            return "BOOL"
    except Exception:
        pass
    return str(dt)


def main():
    try:
        import tensorrt as trt
    except Exception as e:
        print("导入 TensorRT 失败，请确认已安装 TensorRT Python 包。错误：", e)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Inspect TensorRT engine I/O dtypes and shapes (robust)")
    parser.add_argument("engine", nargs="?", default="m.engine", help="Path to .engine file (default: m.engine)")
    parser.add_argument("--plugins", nargs="*", default=None, help="额外要加载的 TensorRT 插件库路径（.dll/.so）")
    parser.add_argument("--trt-lib-dir", default=None, help="TensorRT 安装的 lib 目录（Windows 下会自动加入 DLL 搜索路径）")
    parser.add_argument("--verbose", action="store_true", help="打印更多调试信息")
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    load_trt_plugins(logger, args.plugins, args.trt_lib_dir)

    # 反序列化引擎
    try:
        with open(args.engine, "rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        print(f"反序列化引擎失败: {args.engine} 错误：{e}")
        print("排查建议：\n- 确认 TensorRT Python 版本与 trtexec 一致\n- 使用 --trt-lib-dir 指向 TensorRT/lib 目录\n- 使用 --plugins 指定 nvinfer_plugin_X.dll 与 nvinfer_X.dll 的绝对路径")
        sys.exit(1)

    if engine is None:
        print("Engine 反序列化返回 None，通常是插件未加载或版本不匹配。")
        sys.exit(2)

    print(f"Engine loaded: {args.engine}")
    print(f"TensorRT version: {trt.__version__}")

    # ——— 使用新 API（TRT 10+），适配 get_io_tensor_name / get_tensor_name 两种命名 ———
    inputs, outputs = [], []
    try:
        n = engine.num_io_tensors  # 存在于你的环境
        for i in range(n):
            try:
                name = engine.get_io_tensor_name(i)  # 某些绑定
            except AttributeError:
                name = engine.get_tensor_name(i)     # 你的绑定提供的名称
            mode = engine.get_tensor_mode(name)
            dtype = engine.get_tensor_dtype(name)
            shape = engine.get_tensor_shape(name)
            entry = dict(
                index=i,
                name=name,
                mode=("INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"),
                dtype=dtype_to_str(trt, dtype),
                shape=tuple(shape),
            )
            (inputs if entry["mode"] == "INPUT" else outputs).append(entry)
        api_used = "new-io-tensors(get_tensor_name)"
    except Exception as e_new:
        if args.verbose:
            print("[Debug] New API failed:", repr(e_new))
        # ——— 回退到旧 API（bindings） ———
        try:
            n = engine.num_bindings
            for i in range(n):
                name = engine.get_binding_name(i)
                is_input = engine.binding_is_input(i)
                dtype = engine.get_binding_dtype(i)
                shape = engine.get_binding_shape(i)
                entry = dict(
                    index=i,
                    name=name,
                    mode=("INPUT" if is_input else "OUTPUT"),
                    dtype=dtype_to_str(trt, dtype),
                    shape=tuple(shape),
                )
                (inputs if is_input else outputs).append(entry)
            api_used = "old-bindings"
        except Exception as e_old:
            if args.verbose:
                print("[Debug] Old API failed:", repr(e_old))
                print("[Debug] dir(engine):", dir(engine))
            print("无法找到可用的 TensorRT 引擎接口（IO tensors 或 bindings）。")
            print("建议：\n- 升级/重装与 trtexec 一致版本的 Python TensorRT\n- 使用 --verbose 打印调试信息并将输出发我协助定位")
            sys.exit(3)

    if args.verbose:
        print(f"[Info] API used: {api_used}")

    # 打印结果
    print("\n== INPUTS ==")
    for t in inputs:
        print(f"[{t['index']}] {t['name']:<24} {t['mode']:<6} dtype={t['dtype']:<6} shape={t['shape']}")

    print("\n== OUTPUTS ==")
    for t in outputs:
        print(f"[{t['index']}] {t['name']:<24} {t['mode']:<6} dtype={t['dtype']:<6} shape={t['shape']}")

    input_dtypes = set(t["dtype"] for t in inputs)
    print("\n== SUMMARY ==")
    if not inputs:
        print("未发现输入绑定。")
    else:
        print("输入 dtype 集合:", ", ".join(sorted(input_dtypes)))
        if "HALF" in input_dtypes:
            print("检测到 FP16 输入绑定（HALF）：本项目的 letterbox 将默认走 FP16。")
        if "FLOAT" in input_dtypes:
            print("检测到 FP32 输入绑定（FLOAT）：本项目的 letterbox 将默认走 FP32。")

if __name__ == "__main__":
    main()
