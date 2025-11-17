#include "ovl_impl.hpp"
#include "pch.hpp"

void Game_overlay::Impl::BeginFrame(){ std::lock_guard<std::mutex> lk(pendingMutex); pending.cmds.clear(); }
void Game_overlay::Impl::EndFrame(){ auto dl=std::make_shared<DrawList>(); { std::lock_guard<std::mutex> lk(pendingMutex); dl->cmds=pending.cmds; }
  std::stable_sort(dl->cmds.begin(), dl->cmds.end(), [](const DrawCmd& a, const DrawCmd& b){ return a.layer < b.layer; });
  std::atomic_store_explicit(&current, dl, std::memory_order_release); }
void Game_overlay::Impl::AddCmd(const DrawCmd& c){ std::lock_guard<std::mutex> lk(pendingMutex); pending.cmds.push_back(c); }
