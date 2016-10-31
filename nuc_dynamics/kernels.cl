kernel void updateVelocity(global double * const veloc,
                           global const double * const masses,
                           global const double * const forces,
                           global const double * const accel,
                           const double tStep, const double r) {
    size_t i = get_global_id(0);

    const double3 old = vload3(i, veloc);
    const double3 new =
        old + 0.5 * tStep * (vload3(i, forces) / masses[i] + r * old - vload3(i, accel));
    vstore3(new, i, veloc);
}

kernel void updateMotion(global double * const coords,
                         global double * const veloc,
                         global double * const accel,
                         global const double * const masses,
                         global const double * const forces,
                         const double tStep, const double r) {
    size_t i = get_global_id(0);

    const double3 old_veloc = vload3(i, veloc);
    const double3 old_coords = vload3(i, coords);
    const double3 new_accel =
        vload3(i, forces) / masses[i] + r * old_veloc;
    const double3 new_coords =
        old_coords + tStep * old_veloc + 0.5 * tStep * tStep * new_accel;
    const double3 new_veloc = old_veloc + tStep * new_accel;

    vstore3(new_accel, i, accel);
    vstore3(new_coords, i, coords);
    vstore3(new_veloc, i, veloc);
}

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

double atomic_fadd(volatile global double *p, double val) {
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
}

kernel void getRepulsiveForce(global const int * const repList,
                              global double * const forces,
                              global const double * const coords,
                              global const double * const radii,
                              const double fConst) {
    size_t i = get_global_id(0);

    int j = repList[i*2+0];
    int k = repList[i*2+1];
    double repDist = radii[j] + radii[k];
    double repDist2 = repDist * repDist;

    const double3 dist = vload3(k, coords) - vload3(j, coords);
    if (any(fabs(dist) > repDist))
        return;
    const double dist2 = dot(dist, dist);
    if (dist2 > repDist2)
        return;

    double dists[3]; // Can't index vector
    vstore3(dist, 0, dists);

    const double rjk = 4 * fConst * (repDist2 - dist2);
    const size_t D = 3;
    for (size_t d = 0; d < D; d++){
        atomic_fadd(&forces[j*D+d],-dists[d] * rjk);
        atomic_fadd(&forces[k*D+d], dists[d] * rjk);
    }
}
