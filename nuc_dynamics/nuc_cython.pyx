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


def runDynamics(ctx, cq, kernels, collider,
                coords_buf, masses_buf, radii_buf, int nCoords,
                restIndices_buf, restLimits_buf, restWeights_buf, int nRest,
                restAmbig_buf, int nAmbig,
                double tRef=1000.0, double tStep=0.001, int nSteps=1000,
                double fConstR=1.0, double fConstD=25.0, double beta=10.0,
                double tTaken=0.0, int printInterval=10000,
                double tot0=20.458):

  if nCoords < 2:
    raise NucCythonError('Too few coodinates')

  cdef int i, j, n, step, nViol

  cdef double d2, dx, dy, dz, ek, rmsd, tStep0, temp, fDist, fRep
  cdef double Langevin_gamma

  tStep0 = tStep * tot0
  beta /= tot0

  (masses, _) = cl.enqueue_map_buffer(
    cq, masses_buf, cl.map_flags.READ,
    0, nCoords, dtype('float64'), is_blocking=False
  )
  (radii, _) = cl.enqueue_map_buffer(
    cq, radii_buf, cl.map_flags.READ,
    0, nCoords, dtype('float64'), is_blocking=False
  )
  (restLimits, _) = cl.enqueue_map_buffer(
    cq, restLimits_buf, cl.map_flags.READ,
    0, (nRest, 2), dtype('float64'), is_blocking=False
  )
  (restIndices, _) = cl.enqueue_map_buffer(
    cq, restIndices_buf, cl.map_flags.READ,
    0, (nRest, 2), dtype('int32'), is_blocking=False
  )

  accel_buf = cl.Buffer(
    ctx, cl.mem_flags.HOST_NO_ACCESS | cl.mem_flags.READ_WRITE,
    nCoords * 3 * dtype('double').itemsize
  )
  forces_buf = cl.Buffer(
    ctx, cl.mem_flags.HOST_NO_ACCESS | cl.mem_flags.READ_WRITE,
    nCoords * 3 * dtype('double').itemsize
  )
  veloc_buf = cl.Buffer(
    ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_WRITE,
    nCoords * 3 * dtype('double').itemsize
  )
  nRep_buf = cl.Buffer(
    ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_WRITE, dtype('int32').itemsize
  )

  cl.enqueue_fill_buffer(
    cq, accel_buf, numpy.zeros(1, dtype='float64'),
    0, nCoords * 3 * dtype('float64').itemsize
  )
  (veloc, _) = cl.enqueue_map_buffer(
    cq, veloc_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
    0, (nCoords, 3), dtype('float64'), is_blocking=False
  )
  cl.wait_for_events([cl.enqueue_barrier(cq)])

  veloc[...] = numpy.random.normal(0.0, 1.0, (nCoords, 3))
  veloc *= sqrt(tRef / getTemp(masses, veloc, nCoords))
  for i, m in enumerate(masses):
    if m == INFINITY:
      veloc[i] = 0

  cdef double t0 = time.time()
  cdef double r # for use as kernel parameter

  e = collider.get_collisions(cq, coords_buf, radii_buf, nRep_buf, None, 0)
  (nRep, _) = cl.enqueue_map_buffer(
    cq, nRep_buf, cl.map_flags.READ,
    0, 1, dtype('int32'), wait_for=[e], is_blocking=True,
  )
  nRepMax = int(nRep[0] * 1.2) # Allocate with some padding
  del nRep
  repList_buf = cl.Buffer(
    ctx, cl.mem_flags.HOST_NO_ACCESS | cl.mem_flags.READ_WRITE,
    nRepMax * 2 * dtype('int32').itemsize
  )
  e = collider.get_collisions(
    cq, coords_buf, radii_buf, nRep_buf, repList_buf, nRepMax, [e]
  )
  (nRep, _) = cl.enqueue_map_buffer(
    cq, nRep_buf, cl.map_flags.READ,
    0, 1, dtype('int32'), wait_for=[e], is_blocking=True,
  )
  cl.wait_for_events([cl.enqueue_barrier(cq)])

  zero_forces = cl.enqueue_fill_buffer(
    cq, forces_buf, numpy.zeros(1, dtype='float64'),
    0, nCoords * 3 * dtype('float64').itemsize
  )
  e = kernels['getRepulsiveForce'](
    cq, (nRep,), None,
    repList_buf, forces_buf, coords_buf, radii_buf, masses_buf, fConstR,
    wait_for=[zero_forces]
  )
  e = kernels['getRestraintForce'](
    cq, (nAmbig-1,), None,
    restIndices_buf, restLimits_buf, restWeights_buf, restAmbig_buf,
    coords_buf, forces_buf, fConstD, 0.5,
    wait_for=[zero_forces]
  )
  cl.wait_for_events([cl.enqueue_barrier(cq)])

  for step in range(nSteps):
    del nRep
    e = collider.get_collisions(
      cq, coords_buf, radii_buf, nRep_buf, repList_buf, nRepMax, [e]
    )
    (nRep, _) = cl.enqueue_map_buffer(
      cq, nRep_buf, cl.map_flags.READ,
      0, 1, dtype('int32'), wait_for=[e], is_blocking=True,
    )
    if nRep[0] > nRepMax:
      nRepMax = int(nRep[0] * 1.2)
      del nRep
      repList_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_NO_ACCESS | cl.mem_flags.READ_WRITE,
        nRepMax * 2 * dtype('int32').itemsize
      )
      e = collider.get_collisions(
        cq, coords_buf, radii_buf, nRep_buf, repList_buf, nRepMax, [e]
      )
      (nRep, _) = cl.enqueue_map_buffer(
        cq, nRep_buf, cl.map_flags.READ,
        0, 1, dtype('int32'), wait_for=[e], is_blocking=True,
      )
    elif nRep[0] < (nRepMax // 2):
      nRepMax = int(nRep[0] * 1.2)
      old_repList_buf = repList_buf
      repList_buf = cl.Buffer(
        ctx, cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.READ_WRITE,
        nRepMax * 2 * dtype('int32').itemsize
      )
      cl.enqueue_copy(
        cq, repList_buf, old_repList_buf,
        byte_count=nRep[0] * 2 * dtype('int32').itemsize
      )
      del old_repList_buf

    r = beta * (tRef / max(getTemp(masses, veloc, nCoords), 0.001) - 1.0)
    del veloc

    e = kernels['updateMotion'](
      cq, (nCoords,), None,
      coords_buf, veloc_buf, accel_buf, masses_buf, forces_buf, tStep0, r,
    )
    (veloc, _) = cl.enqueue_map_buffer(
      cq, veloc_buf, cl.map_flags.READ,
      0, (nCoords, 3), dtype('float64'), wait_for=[e], is_blocking=False
    )
    zero_forces = cl.enqueue_fill_buffer(
      cq, forces_buf, numpy.zeros(1, dtype='float64'),
      0, nCoords * 3 * dtype('float64').itemsize, wait_for=[e]
    )
    e = kernels['getRepulsiveForce'](
      cq, (nRep,), None,
      repList_buf, forces_buf, coords_buf, radii_buf, masses_buf, fConstR,
      wait_for=[zero_forces]
    )
    e = kernels['getRestraintForce'](
      cq, (nAmbig-1,), None,
      restIndices_buf, restLimits_buf, restWeights_buf, restAmbig_buf,
      coords_buf, forces_buf, fConstD, 0.5,
      wait_for=[zero_forces]
    )
    cl.wait_for_events([cl.enqueue_barrier(cq)])

    r = beta * (tRef / max(getTemp(masses, veloc, nCoords), 0.001) - 1.0)
    del veloc

    e = kernels['updateVelocity'](
      cq, (nCoords,), None,
      veloc_buf, masses_buf, forces_buf, accel_buf, tStep0, r,
    )

    (veloc, _) = cl.enqueue_map_buffer(
      cq, veloc_buf, cl.map_flags.READ,
      0, (nCoords, 3), dtype('float64'), wait_for=[e], is_blocking=False
    )
    cl.wait_for_events([cl.enqueue_barrier(cq)])

    if (printInterval > 0) and step % printInterval == 0:
      temp = getTemp(masses, veloc, nCoords)
      (coords, _) = cl.enqueue_map_buffer(
        cq, coords_buf, cl.map_flags.READ,
        0, (nCoords, 3), dtype('float64'), wait_for=[e], is_blocking=False
      )
      cl.wait_for_events([cl.enqueue_barrier(cq)])
      nViol, rmsd = getStats(restIndices, restLimits, coords, nRest)
      del coords

      data = (temp, rmsd, nViol, nRep[0])
      print('temp:%7.2lf  rmsd:%7.2lf  nViol:%5d  nRep:%5d' % data)

    tTaken += tStep

  return tTaken
