import numpy
from numpy import dtype
import pyopencl as cl
import time
from collision.misc import roundUp

BOLTZMANN_K = 0.0019872041

def getTemp(masses, veloc):
    indices = masses != float('inf')
    kin = (masses[indices] * (veloc[indices]**2).sum(axis=1)).sum()

    return kin / (3 * len(masses) * BOLTZMANN_K)


def getStats(indices, limits, coords):
    j, k = indices.T
    identical = j == k
    j = j[~identical]
    k = k[~identical]

    dmin, dmax = limits[~identical].T

    r = numpy.linalg.norm(coords[j] - coords[k], axis=1)
    viol = (dmin - r).clip(min=0) + (r - dmax).clip(min=0)
    return numpy.count_nonzero(viol), numpy.sqrt((viol * viol).sum() / len(indices))


def runDynamics(ctx, cq, kernels, collider,
                coords_buf, masses_buf, radii_buf, nCoords,
                restIndices_buf, restLimits_buf, restWeights_buf, nRest,
                restAmbig_buf, nAmbig,
                tRef=1000.0, tStep=0.001, nSteps=1000, fConstR=1.0, fConstD=25.0,
                beta=10.0, tTaken=0.0, printInterval=10000, tot0=20.458):

  if nCoords < 2:
    raise NucCythonError('Too few coodinates')

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
    nCoords * 4 * dtype('double').itemsize
  )
  forces_buf = cl.Buffer(
    ctx, cl.mem_flags.HOST_NO_ACCESS | cl.mem_flags.READ_WRITE,
    nCoords * 4 * dtype('double').itemsize
  )
  veloc_buf = cl.Buffer(
    ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_WRITE,
    nCoords * 4 * dtype('double').itemsize
  )
  nRep_buf = cl.Buffer(
    ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_WRITE, dtype('int32').itemsize
  )

  cl.enqueue_fill_buffer(
    cq, accel_buf, numpy.zeros(1, dtype='float64'),
    0, nCoords * 4 * dtype('float64').itemsize
  )
  (veloc, _) = cl.enqueue_map_buffer(
    cq, veloc_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
    0, (nCoords, 4), dtype('float64'), is_blocking=False
  )
  cl.wait_for_events([cl.enqueue_barrier(cq)])

  veloc[:, :3] = numpy.random.normal(0.0, 1.0, (nCoords, 3))
  veloc *= numpy.sqrt(tRef / getTemp(masses, veloc[:, :3]))
  veloc[masses == float('inf')] = 0

  t0 = time.time()

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
    0, nCoords * 4 * dtype('float64').itemsize
  )
  e = kernels['getRepulsiveForce'](
    cq, (roundUp(nRep[0], 64),), None,
    repList_buf, forces_buf, coords_buf, radii_buf, masses_buf, fConstR, nRep[0],
    wait_for=[zero_forces]
  )
  e = kernels['getRestraintForce'](
      cq, (roundUp(nAmbig-1, 64),), None,
      restIndices_buf, restLimits_buf, restWeights_buf, restAmbig_buf,
      coords_buf, forces_buf, fConstD, 0.5, nAmbig-1,
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

    r = beta * (tRef / max(getTemp(masses, veloc[:, :3]), 0.001) - 1.0)
    del veloc

    e = kernels['updateMotion'](
      cq, (roundUp(nCoords, 64),), None,
      coords_buf, veloc_buf, accel_buf, masses_buf, forces_buf, tStep0, r, nCoords,
    )
    (veloc, _) = cl.enqueue_map_buffer(
      cq, veloc_buf, cl.map_flags.READ,
      0, (nCoords, 4), dtype('float64'), wait_for=[e], is_blocking=False
    )
    zero_forces = cl.enqueue_fill_buffer(
      cq, forces_buf, numpy.zeros(1, dtype='float64'),
      0, nCoords * 4 * dtype('float64').itemsize, wait_for=[e]
    )
    e = kernels['getRepulsiveForce'](
      cq, (roundUp(nRep[0], 64),), None,
      repList_buf, forces_buf, coords_buf, radii_buf, masses_buf, fConstR, nRep[0],
      wait_for=[zero_forces]
    )
    e = kernels['getRestraintForce'](
      cq, (roundUp(nAmbig-1, 64),), None,
      restIndices_buf, restLimits_buf, restWeights_buf, restAmbig_buf,
      coords_buf, forces_buf, fConstD, 0.5, nAmbig-1,
      wait_for=[zero_forces]
    )
    cl.wait_for_events([cl.enqueue_barrier(cq)])

    r = beta * (tRef / max(getTemp(masses, veloc[:, :3]), 0.001) - 1.0)
    del veloc

    e = kernels['updateVelocity'](
      cq, (roundUp(nCoords, 64),), None,
      veloc_buf, masses_buf, forces_buf, accel_buf, tStep0, r, nCoords,
    )

    (veloc, _) = cl.enqueue_map_buffer(
      cq, veloc_buf, cl.map_flags.READ,
      0, (nCoords, 4), dtype('float64'), wait_for=[e], is_blocking=False
    )
    cl.wait_for_events([cl.enqueue_barrier(cq)])

    if (printInterval > 0) and step % printInterval == 0:
      temp = getTemp(masses, veloc[:, :3])
      (coords, _) = cl.enqueue_map_buffer(
        cq, coords_buf, cl.map_flags.READ,
        0, (nCoords, 4), dtype('float64'), wait_for=[e], is_blocking=False
      )
      cl.wait_for_events([cl.enqueue_barrier(cq)])
      nViol, rmsd = getStats(restIndices, restLimits, coords[:, :3])
      del coords

      data = (temp, rmsd, nViol, nRep[0])
      print('temp:%7.2lf  rmsd:%7.2lf  nViol:%5d  nRep:%5d' % data)

    tTaken += tStep

  return tTaken
