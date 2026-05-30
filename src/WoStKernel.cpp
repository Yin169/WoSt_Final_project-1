#include "WoStKernel.hpp"
#include <algorithm>
#include <cmath>

namespace wost {

WoStKernel::WoStKernel(const WoStGeometryBackend& inner, const CubeOuterBoundary& outer)
    : inner_(inner), outer_(outer) {}

bool WoStKernel::InDomain(const vec3& x) const {
    return outer_.IsInside(x) && !inner_.IsInside(x);
}

BoundaryPoint WoStKernel::makeCubeBP(const vec3& origin, const vec3& dir, float t, const vec3& normal) {
    BoundaryPoint bp;
    bp.position = madd(origin, dir, t);
    bp.normal   = normal;
    bp.dist     = t;
    bp.triIdx   = ~0u;
    return bp;
}

// ===========================================================================
// (1) SolveLaplace -> 直接复用 SolvePoisson 降低维护成本
// ===========================================================================
WalkResult WoStKernel::SolveLaplace(const vec3&          x0,
                                    const DirichletFn&   g_inner,
                                    const NeumannPredFn& is_inner_neumann,
                                    const NeumannFn&     h_inner,
                                    const DirichletFn&   g_outer,
                                    const WoStParams&    params) const
{
    return SolvePoisson(x0, g_inner, is_inner_neumann, h_inner, g_outer,
                        [](const vec3& x) { (void)x; return 0.f; }, params);
}

// ===========================================================================
// (2) SolvePoisson
// ===========================================================================
WalkResult WoStKernel::SolvePoisson(const vec3&          x0,
                                    const DirichletFn&   g_inner,
                                    const NeumannPredFn& is_inner_neumann,
                                    const NeumannFn&     h_inner,
                                    const DirichletFn&   g_outer,
                                    const SourceFn&      f,
                                    const WoStParams&    params) const
{
    WalkResult result;
    double sumV = 0.0, sumV2 = 0.0;
    int sumSteps = 0;
    FastRNG rng(static_cast<uint32_t>(params.seed));

    for (int s = 0; s < params.numSamples; ++s) {
        vec3  x    = x0;
        float acc  = 0.f;
        int   steps = 0;
        bool  done  = false;

        for (int step = 0; step < params.maxSteps; ++step) {
            ++steps;
            float R_inner_approx = inner_.FastBoundaryDistance(x);
            float R_outer        = outer_.FastStarRadius(x);
            float R_approx       = std::min(R_inner_approx, R_outer);

            if (R_approx < params.eps * 2.0f) {
                // ── 近边界：精确计算 star radius ──────────────────────────
                BoundaryPoint bndBP_inner, silBP_inner;
                float R_inner = inner_.StarRadius(x, bndBP_inner, silBP_inner);
                float R       = std::min(R_inner, R_outer);
                bool  outerCloser = (R_outer <= R_inner);

                float distToActualBoundary = outerCloser ? R_outer : bndBP_inner.dist;

                if (distToActualBoundary < params.eps) {
                    if (outerCloser) {
                        BoundaryPoint bndBP_outer;
                        outer_.ClosestPoint(x, bndBP_outer);
                        acc += g_outer(bndBP_outer);
                        done = true;
                        break;
                    } else {
                        if (is_inner_neumann(bndBP_inner)) {
                            // 🚀 修复核心：定义一个宏观跳跃距离，彻底脱离边界引力
                            float jump_dist = std::max(params.eps * 100.0f, R_outer * 0.05f);
                            
                            // 差分近似积分（注意乘以的是跳跃距离 jump_dist，而不是 eps）
                            acc += h_inner(bndBP_inner) * jump_dist;
                            
                            // 沿法线推回域内一个宏观距离
                            x = madd(bndBP_inner.position, bndBP_inner.normal, jump_dist);
                        } else {
                            // Dirichlet 边界直接吸收终止
                            acc += g_inner(bndBP_inner);
                            done = true;
                            break;
                        }
                    }
                }

                if (!done) {
                    vec3     dir = sampleUnitSphere(rng);
                    float    t_inner;
                    vec3     n_inner;
                    uint32_t prim_inner;
                    bool hit = inner_.IntersectRay(x, dir, R, t_inner, n_inner, prim_inner);

                    // ✅ Bug 2 修正：源项体积积分始终使用完整控制球半径 R
                    acc -= (R * R / 6.f) * f(x);

                    if (hit) {
                        BoundaryPoint bp = makeBP(x, dir, t_inner, n_inner, prim_inner);
                        if (is_inner_neumann(bp)) {
                            // ✅ Bug 1 修正：无偏 Neumann 积分面积元权重
                            float cosTheta = std::max(1e-4f, std::abs(dot3(dir, bp.normal)));
                            acc += h_inner(bp) * t_inner * (1.0f - t_inner / R) / cosTheta;
                            
                            // 🚀 修复核心：计算反射方向，并走完剩余的射线距离
                            vec3 reflected_dir = dir - madd(bp.normal, 2.0f * dot3(dir, bp.normal), 0.0f);
                            float remaining_dist = R - t_inner;
                            
                            // 将粒子移动到反射后的位置，并沿法线加上 eps 防止浮点误差导致的自交
                            vec3 new_pos = madd(bp.position, reflected_dir, remaining_dist);
                            x = madd(new_pos, bp.normal, params.eps);
                        } else {
                            acc += g_inner(bp);
                            done = true;
                            break;
                        }
                    } else {
                        x = madd(x, dir, R);
                    }
                }
            } else {
                // ── 远离边界：使用快速近似 star radius ───────────────────────
                float R   = R_approx;
                vec3  dir = sampleUnitSphere(rng);
                float    t_inner;
                vec3     n_inner;
                uint32_t prim_inner;
                bool hit = inner_.IntersectRay(x, dir, R, t_inner, n_inner, prim_inner);

                // ✅ Bug 2 修正：源项体积积分始终使用完整控制球半径 R
                acc -= (R * R / 6.f) * f(x);

                if (hit) {
                    BoundaryPoint bp = makeBP(x, dir, t_inner, n_inner, prim_inner);
                    if (is_inner_neumann(bp)) {
                        // ✅ Bug 1 修正：无偏 Neumann 积分面积元权重
                        float cosTheta = std::max(1e-4f, std::abs(dot3(dir, bp.normal)));
                        acc += h_inner(bp) * t_inner * (1.0f - t_inner / R) / cosTheta;
                        
                        // 🚀 修复核心：计算反射方向，并走完剩余的射线距离
                        vec3 reflected_dir = dir - madd(bp.normal, 2.0f * dot3(dir, bp.normal), 0.0f);
                        float remaining_dist = R - t_inner;
                        
                        // 将粒子移动到反射后的位置，并沿法线加上 eps 防止浮点误差导致的自交
                        vec3 new_pos = madd(bp.position, reflected_dir, remaining_dist);
                        x = madd(new_pos, bp.normal, params.eps);
                    } else {
                        acc += g_inner(bp);
                        done = true;
                        break;
                    }
                } else {
                    x = madd(x, dir, R);
                }
            }
        } // step loop

        // 步数超限时的保底处理
        if (!done) {
            BoundaryPoint bp_i, bp_o;
            float d_i = inner_.ClosestPoint(x, bp_i);
            float d_o = outer_.ClosestPoint(x, bp_o);
            acc += (d_o <= d_i) ? g_outer(bp_o) : g_inner(bp_i);
            result.anyDiverged = true;
        }

        sumV     += acc;
        sumV2    += static_cast<double>(acc) * acc;
        sumSteps += steps;
    }

    finalise(result, sumV, sumV2, sumSteps, params.numSamples);
    return result;
}

} // namespace wost