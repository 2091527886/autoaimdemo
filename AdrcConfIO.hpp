#pragma once
#include <string>
class Config;

// 仅解析 ADRC 相关键（adrc_*）以及与输出量化精度相关的少量键（如 aim_residual_carry），将值写入 Config 对应字段
// 返回：存在并成功解析至少一个键返回 true；文件不存在或无有效键返回 false
bool LoadAdrcConfIfPresent(const std::string& path, Config& c);
