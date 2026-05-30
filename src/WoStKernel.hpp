#ifndef DUAL_BOUNDARY_KERNEL_HPP
#define DUAL_BOUNDARY_KERNEL_HPP

#include "WoStGeometryBackend.hpp"
#include "CubeOuterBoundary.hpp"
#include "utils.hpp"

namespace wost {

class WoStKernel {
public:
    WoStKernel(const WoStGeometryBackend& inner, const CubeOuterBoundary& outer);

    bool InDomain(const vec3& x) const;

    // 更新：增加 Neumann 边界相关的回调参数
    WalkResult SolveLaplace(const vec3&          x,
                            const DirichletFn&   g_inner,
                            const NeumannPredFn& is_inner_neumann,
                            const NeumannFn&     h_inner,
                            const DirichletFn&   g_outer,
                            const WoStParams&    p = {}) const;

    // 更新：增加 Neumann 边界相关的回调参数
    WalkResult SolvePoisson(const vec3&          x,
                            const DirichletFn&   g_inner,
                            const NeumannPredFn& is_inner_neumann,
                            const NeumannFn&     h_inner,
                            const DirichletFn&   g_outer,
                            const SourceFn&      f,
                            const WoStParams&    p = {}) const;

private:
    static BoundaryPoint makeCubeBP(const vec3& origin,
                                    const vec3& dir,
                                    float       t,
                                    const vec3& normal);

    const WoStGeometryBackend& inner_;
    const CubeOuterBoundary&   outer_;
};

} // namespace wost

#endif // DUAL_BOUNDARY_KERNEL_HPP