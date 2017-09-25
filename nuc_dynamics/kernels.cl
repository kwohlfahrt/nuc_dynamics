#define CAT_HELPER(X,Y) X##Y
#define CAT(X,Y) CAT_HELPER(X,Y)

#define DTYPE double
#define VTYPE CAT(DTYPE,3)
// #if DTYPE==double doesn't seem to work
#define ATOM_ADD atom_fadd

kernel void updateVelocity(global VTYPE * const veloc,
                           global const DTYPE * const masses,
                           global const VTYPE * const forces,
                           global const VTYPE * const accel,
                           const DTYPE tStep, const DTYPE r,
                           const uint n) {
    if (get_global_id(0) >= n)
        return;
    const size_t i = get_global_id(0);

    veloc[i] += ((DTYPE) 0.5) * tStep * (forces[i] / masses[i] + r * veloc[i] - accel[i]);
}

kernel void updateMotion(global VTYPE * const coords,
                         global VTYPE * const veloc,
                         global VTYPE * const accel,
                         global const DTYPE * const masses,
                         global const VTYPE * const forces,
                         const DTYPE tStep, const DTYPE r,
                         const uint n) {
    if (get_global_id(0) >= n)
        return;
    const size_t i = get_global_id(0);

    accel[i] = forces[i] / masses[i] + r * veloc[i];
    coords[i] += tStep * veloc[i] + ((DTYPE) 0.5) * tStep * tStep * accel[i];
    veloc[i] += tStep * accel[i];
}

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

double atom_fadd(volatile global double *p, double val) {
    // https://streamcomputing.eu/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
    union {
        unsigned long u;
        double f;
    } new, prev, current;
    current.f = *p;

    do {
        prev.f = current.f;
        new.f = prev.f + val;
        current.u = atom_cmpxchg((volatile global unsigned long *) p, prev.u, new.u);
    } while (current.u != prev.u);
    return current.f;
}

float atomic_fadd(volatile global float *p, float val) {
    // https://streamcomputing.eu/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
    union {
        unsigned int u;
        float f;
    } new, prev, current;
    current.f = *p;

    do {
        prev.f = current.f;
        new.f = prev.f + val;
        current.u = atomic_cmpxchg((volatile global unsigned int *) p, prev.u, new.u);
    } while (current.u != prev.u);
    return current.f;
}

kernel void getRepulsiveForce(global const int * const repList,
                              global DTYPE * const forces,
                              global const VTYPE * const coords,
                              global const DTYPE * const radii,
                              global const DTYPE * const masses,
                              const DTYPE fConst, const uint n) {
    if (get_global_id(0) >= n)
        return;
    const size_t i = get_global_id(0);

    const int j = repList[i*2+0];
    const int k = repList[i*2+1];
    if (isinf(masses[j]) || isinf(masses[k]))
        return;
    const DTYPE repDist = radii[j] + radii[k];
    const DTYPE repDist2 = repDist * repDist;

    const VTYPE dist = coords[k] - coords[j];
    if (any(fabs(dist) > repDist))
        return;
    const DTYPE dist2 = dot(dist, dist);
    if (dist2 > repDist2)
        return;

    DTYPE dists[3]; // Can't index vector
    vstore3(dist, 0, dists);

    const DTYPE rjk = 4 * fConst * (repDist2 - dist2);
    for (size_t d = 0; d < 3; d++){
        ATOM_ADD(&forces[j*4+d],-dists[d] * rjk);
        ATOM_ADD(&forces[k*4+d], dists[d] * rjk);
    }
}

kernel void getRestraintForce(global const uint2 * const restIndices,
                              global const CAT(DTYPE,2) * const restLimits,
                              global const DTYPE * const restWeights,
                              global const uint * const restAmbig,
                              global const VTYPE * const coords,
                              global DTYPE * const forces,
                              const DTYPE fConst, const DTYPE switchRatio,
                              const unsigned int exp, const uint n) {
    if (get_global_id(0) >= n)
        return;
    const size_t i = restAmbig[get_global_id(0)];
    const size_t nAmbig = restAmbig[get_global_id(0) + 1] - i;

    DTYPE r_inv = 0.0;
    for (size_t n = 0; n < nAmbig; n++){
        const uint2 idxs = restIndices[i+n];
        if (idxs.s0 == idxs.s1)
            continue;
        const DTYPE dist = length(coords[idxs.s0] - coords[idxs.s1]);
        r_inv += pown(dist, -exp);
    }
    const DTYPE r = powr(r_inv, ((DTYPE) -1.0)/exp);

    const CAT(DTYPE,2) limits = restLimits[i];
    const DTYPE distSwitch = limits.s1 * switchRatio;

    const DTYPE rjk = fConst *
        (fmax(limits.s0 - r, (DTYPE) 0.) - clamp(r - limits.s1, (DTYPE) 0., distSwitch));

    for (size_t n = 0; n < nAmbig; n++){
        const uint2 idxs = restIndices[i+n];
        if (idxs.s0 == idxs.s1)
            continue;

        const VTYPE diff = coords[idxs.s0] - coords[idxs.s1];
        const DTYPE dist = length(diff);
        const DTYPE s2 = max(dot(dist, dist), (DTYPE) 1e-8);
        const DTYPE t = rjk * pown(r, exp+1) / pown(dist, exp+2) * restWeights[i+n];
        DTYPE f[3];
        vstore3(t * diff, 0, f);
        for (size_t d = 0; d < 3; d++){
            ATOM_ADD(&forces[idxs.s0*4+d], f[d]);
            ATOM_ADD(&forces[idxs.s1*4+d],-f[d]);
        }
    }
}
