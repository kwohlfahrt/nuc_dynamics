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
