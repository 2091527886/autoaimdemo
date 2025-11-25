#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"

// Only precise mouse movement with residual accumulation retained.

void Aim::mv_precise(double wx, double wy, double &ms) {
  auto t0 = (c.timing ? tn() : tp{});
  double want_x = wx + mv_residual_.x;
  double want_y = wy + mv_residual_.y;
  int dx = (int)LR(want_x);
  int dy = (int)LR(want_y);
  if (dx || dy) {
    mouse.Move(dx, dy);
  }
  double loss = std::hypot(want_x - dx, want_y - dy);
  if (loss > 0) { mvq_loss_sum_ += loss; ++mvq_events_; }
  double carry = CLP(c.aim_residual_carry, 0.0, 1.0);
  mv_residual_.x = (want_x - dx) * carry;
  mv_residual_.y = (want_y - dy) * carry;
  auto t1 = (c.timing ? tn() : t0);
  ms = c.timing ? msd(t1 - t0).count() : 0.0;
}
