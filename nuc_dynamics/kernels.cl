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
