from libc.math cimport exp, abs, sqrt, ceil, pow
from numpy cimport ndarray, double_t, int_t, dtype
from numpy.math cimport INFINITY
import numpy, time, pyopencl as cl

BOLTZMANN_K = 0.0019872041

#ctypedef int_t   int
#ctypedef double_t double

class NucCythonError(Exception):

  def __init__(self, err):

    Exception.__init__(self, err)


cdef int getRepulsionList(ndarray[int,   ndim=2] repList,
                          ndarray[double, ndim=2] coords,
                          ndarray[double, ndim=1] repDists,
                          ndarray[double, ndim=1] radii,
                          ndarray[double, ndim=1] masses):

  cdef int i, j
  cdef int n = 0
  cdef double dx, dy, dz, d2
  cdef double distLim
  cdef double distLim2

  for i in range(len(coords)-2):
    if masses[i] == INFINITY:
      continue

    for j in range(i+2, len(coords)):
      if masses[j] == INFINITY:
        continue

      distLim = repDists[i] + radii[i] + repDists[j] + radii[j]
      distLim2 = distLim * distLim

      dx = coords[i,0] - coords[j,0]
      if abs(dx) > distLim:
        continue

      dy = coords[i,1] - coords[j,1]
      if abs(dy) > distLim:
        continue

      dz = coords[i,2] - coords[j,2]
      if abs(dz) > distLim:
        continue

      d2 = dx*dx + dy*dy + dz*dz

      if d2 > distLim2:
        continue

      # If max is exceeded, array will be resized and recalculated
      if n < len(repList):
        repList[n,0] = i
        repList[n,1] = j

      n += 1

  return n


cpdef double getTemp(ndarray[double, ndim=1] masses,
                     ndarray[double, ndim=2] veloc,
                     int nCoords):
  cdef int i
  cdef double kin = 0.0

  for i in range(nCoords):
    if masses[i] == INFINITY:
      continue
    kin += masses[i] * (veloc[i,0]*veloc[i,0] + veloc[i,1]*veloc[i,1] + veloc[i,2]*veloc[i,2])

  return kin / (3 * nCoords * BOLTZMANN_K)


def getStats(ndarray[int,   ndim=2] restIndices,
             ndarray[double, ndim=2] restLimits,
             ndarray[double, ndim=2] coords,
             int nRest):

  cdef int i, nViol = 0
  cdef int j, k
  cdef double viol, dmin, dmax, dx, dy, dz, r, s = 0

  for i in range(nRest):
    j = restIndices[i,0]
    k = restIndices[i,1]

    if j == k:
      continue

    dmin = restLimits[i,0]
    dmax = restLimits[i,1]

    dx = coords[j,0] - coords[k,0]
    dy = coords[j,1] - coords[k,1]
    dz = coords[j,2] - coords[k,2]
    r = sqrt(dx*dx + dy*dy + dz*dz)

    if r < dmin:
      viol = dmin - r
      nViol += 1

    elif r > dmax:
      viol = r - dmax
      nViol += 1

    else:
      viol = 0

    s += viol * viol

  return nViol, sqrt(s/nRest)


cpdef double getRestraintForce(ndarray[double, ndim=2] forces,
                              ndarray[double, ndim=2] coords,
                              ndarray[int,   ndim=2] restIndices,
                              ndarray[double, ndim=2] restLimits,
                              ndarray[double, ndim=1] restWeight,
                              ndarray[int, ndim=1] restAmbig,
                              double fConst, double exponent=2.0,
                              double switchRatio=0.5, double asymptote=1.0):

  cdef int i, j, k, n, m, nAmbig
  cdef double a, b, d, dmin, dmax, dx, dy, dz, distSwitch
  cdef double r, r2, s2, rjk, ujk, force = 0, t

  for m in range(len(restAmbig) - 1):
    nAmbig = restAmbig[m+1] - restAmbig[m]
    i = restAmbig[m]
    r2 = 0.0

    for n in range(nAmbig):
      j = restIndices[i+n,0]
      k = restIndices[i+n,1]

      if j == k:
        continue

      dx = coords[j,0] - coords[k,0]
      dy = coords[j,1] - coords[k,1]
      dz = coords[j,2] - coords[k,2]
      r = dx*dx + dy*dy + dz*dz
      r2 += 1.0 / (r * r)

    if r2 <= 0:
      continue

    r2 = 1.0 / sqrt(r2)

    dmin = restLimits[i,0]
    dmax = restLimits[i,1]
    distSwitch = dmax * switchRatio

    if r2 < dmin*dmin:
      r2 = max(r2, 1e-8)
      d = dmin - sqrt(r2)
      ujk = fConst * d * d
      rjk = fConst * exponent * d

    elif dmin*dmin <= r2 <= dmax*dmax:
      ujk = rjk = 0
      r = 1.0

    elif dmax*dmax < r2 <= (dmax+distSwitch) * (dmax+distSwitch):
      d = sqrt(r2) - dmax
      ujk = fConst * d * d
      rjk = - fConst * exponent * d

    else: # (dmax+distSwitch) ** 2 < r2
      b = distSwitch * distSwitch * distSwitch * exponent * (asymptote - 1)
      a = distSwitch * distSwitch * (1 - 2*asymptote*exponent + exponent)

      d = sqrt(r2) - dmax
      ujk = fConst * (a + asymptote*distSwitch*exponent*d + b/d)
      rjk = - fConst * (asymptote*distSwitch*exponent - b/(d*d))

    force += ujk

    for n in range(nAmbig):
      j = restIndices[i+n,0]
      k = restIndices[i+n,1]

      if j == k:
        continue

      dx = coords[j,0] - coords[k,0]
      dy = coords[j,1] - coords[k,1]
      dz = coords[j,2] - coords[k,2]

      s2 = max(dx*dx + dy*dy + dz*dz, 1e-08)
      t = rjk * pow(r2, 2.5) / (s2 * s2 * s2) * restWeight[i+n]

      dx *= t
      dy *= t
      dz *= t

      forces[j,0] += dx
      forces[k,0] -= dx
      forces[j,1] += dy
      forces[k,1] -= dy
      forces[j,2] += dz
      forces[k,2] -= dz

  return force


def runDynamics(ctx, cq, kernels,
                coords_buf, masses_buf, radii_buf, repDists_buf, int nCoords,
                ndarray[int, ndim=2] restIndices,
                ndarray[double, ndim=2] restLimits,
                ndarray[double, ndim=1] restWeight,
                ndarray[int, ndim=1] restAmbig,
                double tRef=1000.0, double tStep=0.001, int nSteps=1000,
                double fConstR=1.0, double fConstD=25.0, double beta=10.0,
                double tTaken=0.0, int printInterval=10000,
                double tot0=20.458):

  cdef int nRest = len(restIndices)

  if nCoords < 2:
    raise NucCythonError('Too few coodinates')

  indices = set(restIndices.ravel())
  if min(indices) < 0:
    raise NucCythonError('Restraint index negative')

  if max(indices) >= nCoords:
    data = (max(indices), nCoords)
    raise NucCythonError('Restraint index "%d" out of bounds (> %d)' % data)

  if nRest != len(restLimits):
    raise NucCythonError('Number of restraint index pairs does not match number of restraint limits')

  cdef int i, j, n, step, nViol, nRep = 0

  cdef double d2, dx, dy, dz, ek, rmsd, tStep0, temp, fDist, fRep
  cdef double Langevin_gamma

  tStep0 = tStep * tot0
  beta /= tot0

  (coords, _) = cl.enqueue_map_buffer(
    cq, coords_buf, cl.map_flags.READ,
    0, (nCoords, 3), dtype('float64'), is_blocking=False
  )
  (masses, _) = cl.enqueue_map_buffer(
    cq, masses_buf, cl.map_flags.READ,
    0, nCoords, dtype('float64'), is_blocking=False
  )
  (radii, _) = cl.enqueue_map_buffer(
    cq, radii_buf, cl.map_flags.READ,
    0, nCoords, dtype('float64'), is_blocking=False
  )
  (repDists, _) = cl.enqueue_map_buffer(
    cq, repDists_buf, cl.map_flags.READ,
    0, nCoords, dtype('float64'), is_blocking=False
  )

  accel_buf = cl.Buffer(
    ctx, cl.mem_flags.HOST_NO_ACCESS | cl.mem_flags.READ_WRITE,
    nCoords * 3 * dtype('double').itemsize
  )
  forces_buf = cl.Buffer(
    ctx, cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.READ_WRITE,
    nCoords * 3 * dtype('double').itemsize
  )
  veloc_buf = cl.Buffer(
    ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_WRITE,
    nCoords * 3 * dtype('double').itemsize
  )

  cl.enqueue_fill_buffer(
    cq, accel_buf, numpy.zeros(1, dtype='float64'),
    0, nCoords * 3 * dtype('float64').itemsize
  )
  cl.enqueue_fill_buffer(
    cq, forces_buf, numpy.zeros(1, dtype='float64'),
    0, nCoords * 3 * dtype('float64').itemsize
  )
  (veloc, _) = cl.enqueue_map_buffer(
    cq, veloc_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
    0, (nCoords, 3), dtype('float64'), is_blocking=False
  )
  cl.wait_for_events([cl.enqueue_barrier(cq)])

  cdef ndarray[double, ndim=1] deltaLim = repDists * repDists
  cdef ndarray[double, ndim=2] coordsPrev = numpy.array(coords)

  veloc[...] = numpy.random.normal(0.0, 1.0, (nCoords, 3))
  veloc *= sqrt(tRef / getTemp(masses, veloc, nCoords))
  for i, m in enumerate(masses):
    if m == INFINITY:
      veloc[i] = 0

  cdef double t0 = time.time()
  cdef double r # for use as kernel parameter

  nRep = getRepulsionList(
    numpy.empty((0, 2), dtype='int32'), coords, repDists, radii, masses
  )
  nRepMax = int(nRep * 1.2) # Allocate with some padding
  repList_buf = cl.Buffer(
    ctx, cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.READ_WRITE,
    nRepMax * 2 * dtype('int32').itemsize
  )
  (repList, _) = cl.enqueue_map_buffer(
    cq, repList_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
    0, (nRepMax, 2), dtype('int32'), is_blocking=True
  )
  nRep = getRepulsionList(repList, coords, repDists, radii, masses)

  del repList
  e = kernels['getRepulsiveForce'](
    cq, (nRep,), None,
    repList_buf, forces_buf, coords_buf, radii_buf, fConstR,
  )
  (forces, _) = cl.enqueue_map_buffer(
    cq, forces_buf, cl.map_flags.READ | cl.map_flags.WRITE,
    0, (nCoords, 3), dtype('float64'), wait_for=[e], is_blocking=True
  )
  fDist = getRestraintForce(forces, coords, restIndices, restLimits,
                            restWeight, restAmbig, fConstD)

  for step in range(nSteps):
    for i in range(nCoords):
      dx = coords[i,0] - coordsPrev[i,0]
      dy = coords[i,1] - coordsPrev[i,1]
      dz = coords[i,2] - coordsPrev[i,2]
      if dx*dx + dy*dy + dz*dz > deltaLim[i]:
        (repList, _) = cl.enqueue_map_buffer(
          cq, repList_buf, cl.map_flags.READ | cl.map_flags.WRITE,
          0, (nRepMax, 2), dtype('int32'), wait_for=[e], is_blocking=True
        )
        nRep = getRepulsionList(repList, coords, repDists, radii, masses)
        del repList
        if nRep > nRepMax:
          nRepMax = int(nRep * 1.2)
          repList_buf = cl.Buffer(ctx, cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.READ_WRITE,
                                  nRepMax * 2 * dtype('int32').itemsize)
          (repList, _) = cl.enqueue_map_buffer(
            cq, repList_buf, cl.map_flags.READ | cl.map_flags.WRITE,
            0, (nRepMax, 2), dtype('int32'), is_blocking=True
          )
          nRep = getRepulsionList(repList, coords, repDists, radii, masses)
          del repList
        elif nRep < (nRepMax // 2):
          nRepMax = int(nRep * 1.2)
          old_repList_buf = repList_buf
          repList_buf = cl.Buffer(ctx, cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.READ_WRITE,
                                  nRepMax * 2 * dtype('int32').itemsize)
          cl.enqueue_copy(cq, repList_buf, old_repList_buf,
                          byte_count=nRep * 2 * dtype('int32').itemsize)
          del old_repList_buf

        for i in range(nCoords):
          coordsPrev[i,0] = coords[i,0]
          coordsPrev[i,1] = coords[i,1]
          coordsPrev[i,2] = coords[i,2]
        break # Already re-calculated, no need to check more

    r = beta * (tRef / max(getTemp(masses, veloc, nCoords), 0.001) - 1.0)
    del coords, veloc, forces

    e = kernels['updateMotion'](
      cq, (nCoords,), None,
      coords_buf, veloc_buf, accel_buf, masses_buf, forces_buf, tStep0, r,
    )
    (veloc, _) = cl.enqueue_map_buffer(
      cq, veloc_buf, cl.map_flags.READ,
      0, (nCoords, 3), dtype('float64'), wait_for=[e], is_blocking=False
    )
    e = cl.enqueue_fill_buffer(
      cq, forces_buf, numpy.zeros(1, dtype='float64'),
      0, nCoords * 3 * dtype('float64').itemsize, wait_for=[e]
    )
    e = kernels['getRepulsiveForce'](
      cq, (nRep,), None,
      repList_buf, forces_buf, coords_buf, radii_buf, fConstR,
      wait_for=[e]
    )

    (coords, _) = cl.enqueue_map_buffer(
      cq, coords_buf, cl.map_flags.READ,
      0, (nCoords, 3), dtype('float64'), wait_for=[e], is_blocking=False
    )
    (forces, _) = cl.enqueue_map_buffer(
      cq, forces_buf, cl.map_flags.READ | cl.map_flags.WRITE,
      0, (nCoords, 3), dtype('float64'), wait_for=[e], is_blocking=False
    )
    cl.wait_for_events([cl.enqueue_barrier(cq)])

    fDist = getRestraintForce(forces, coords, restIndices, restLimits,
                              restWeight, restAmbig, fConstD)

    r = beta * (tRef / max(getTemp(masses, veloc, nCoords), 0.001) - 1.0)
    del veloc, forces

    e = kernels['updateVelocity'](
      cq, (nCoords,), None,
      veloc_buf, masses_buf, forces_buf, accel_buf, tStep0, r,
    )

    (forces, _) = cl.enqueue_map_buffer(
      cq, forces_buf, cl.map_flags.READ | cl.map_flags.WRITE,
      0, (nCoords, 3), dtype('float64'), wait_for=[e], is_blocking=False
    )
    (veloc, _) = cl.enqueue_map_buffer(
      cq, veloc_buf, cl.map_flags.READ,
      0, (nCoords, 3), dtype('float64'), wait_for=[e], is_blocking=False
    )
    cl.wait_for_events([cl.enqueue_barrier(cq)])

    if (printInterval > 0) and step % printInterval == 0:
      temp = getTemp(masses, veloc, nCoords)
      nViol, rmsd = getStats(restIndices, restLimits, coords, nRest)

      data = (temp, fDist, rmsd, nViol, nRep)
      print('temp:%7.2lf  fDist:%7.2lf  rmsd:%7.2lf  nViol:%5d  nRep:%5d' % data)

    tTaken += tStep

  return tTaken
