kernel void updateVelocity(global double3 * const veloc,
                           global const double * const masses,
                           global const double3 * const forces,
                           global const double3 * const accel,
                           const double tStep, const double r) {
    const size_t i = get_global_id(0);

    veloc[i] += 0.5 * tStep * (forces[i] / masses[i] + r * veloc[i] - accel[i]);
}

kernel void updateMotion(global double3 * const coords,
                         global double3 * const veloc,
                         global double3 * const accel,
                         global const double * const masses,
                         global const double3 * const forces,
                         const double tStep, const double r) {
    const size_t i = get_global_id(0);

    accel[i] = forces[i] / masses[i] + r * veloc[i];
    coords[i] += tStep * veloc[i] + 0.5 * tStep * tStep * accel[i];
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

kernel void getRepulsiveForce(global const int * const repList,
                              global double * const forces,
                              global const double3 * const coords,
                              global const double * const radii,
                              global const double * const masses,
                              const double fConst) {
    const size_t i = get_global_id(0);

    const int j = repList[i*2+0];
    const int k = repList[i*2+1];
    if (isinf(masses[j]) || isinf(masses[k]))
        return;
    const double repDist = radii[j] + radii[k];
    const double repDist2 = repDist * repDist;

    const double3 dist = coords[k] - coords[j];
    if (any(fabs(dist) > repDist))
        return;
    const double dist2 = dot(dist, dist);
    if (dist2 > repDist2)
        return;

    double dists[3]; // Can't index vector
    vstore3(dist, 0, dists);

    const double rjk = 4 * fConst * (repDist2 - dist2);
    for (size_t d = 0; d < 3; d++){
        atom_fadd(&forces[j*4+d],-dists[d] * rjk);
        atom_fadd(&forces[k*4+d], dists[d] * rjk);
    }
}

kernel void getRestraintForce(global const uint2 * const restIndices,
                              global const double2 * const restLimits,
                              global const double * const restWeights,
                              global const uint * const restAmbig,
                              global const double3 * const coords,
                              global double * const forces,
                              const double fConst, const double switchRatio) {
    const size_t i = restAmbig[get_global_id(0)];
    const size_t nAmbig = restAmbig[get_global_id(0) + 1] - i;

    double r_4 = 0.0;
    for (size_t n = 0; n < nAmbig; n++){
        const uint2 idxs = restIndices[i+n];
        if (idxs.s0 == idxs.s1)
            continue;
        const double3 dist = coords[idxs.s0] - coords[idxs.s1];
        const double r2 = dot(dist, dist);
        r_4 += 1.0 / (r2 * r2); // actually 1 / r4
    }
    const double r = rsqrt(sqrt(r_4));

    const double2 limits = restLimits[i];
    const double distSwitch = limits.s1 * switchRatio;

    const double rjk = fConst *
        (fmax(limits.s0 - r, 0.) - clamp(r - limits.s1, 0., distSwitch));

    for (size_t n = 0; n < nAmbig; n++){
        const uint2 idxs = restIndices[i+n];
        if (idxs.s0 == idxs.s1)
            continue;

        const double3 dist = coords[idxs.s0] - coords[idxs.s1];
        double dists[3];
        vstore3(dist, 0, dists);
        const double s2 = max(dot(dist, dist), 1e-8);
        const double t = rjk * pown(r, 5) / (s2 * s2 * s2) * restWeights[i+n];
        for (size_t d = 0; d < 3; d++){
            const double dist = t * dists[d];
            atom_fadd(&forces[idxs.s0*4+d], dist);
            atom_fadd(&forces[idxs.s1*4+d],-dist);
        }
    }
}