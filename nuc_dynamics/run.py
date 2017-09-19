import numpy
from numpy import dtype
import pyopencl as cl
import time
from collision.misc import roundUp

fp_type = dtype('float64')

BOLTZMANN_K = 0.0019872041

def std(x):
    return numpy.sqrt(((x - x.mean(axis=0)) ** 2).sum(axis=1).mean())


def divergence(coords, forces):
    return ((coords - coords.mean(axis=0))[:, :3] *
            (forces - forces.mean(axis=0))[:, :3]).sum(axis=1).mean()

def getTemp(masses, veloc):
    indices = masses != float('inf')
    kin = (masses[indices] * (veloc[indices]**2).sum(axis=1)).sum()

    return kin / (3 * len(masses) * BOLTZMANN_K)


def getStats(indices, limits, weights, coords):
    j, k = indices.T
    identical = j == k
    j = j[~identical]
    k = k[~identical]

    dmin, dmax = limits[~identical].T

    r = numpy.linalg.norm(coords[j] - coords[k], axis=1)
    viol = (dmin - r).clip(min=0) + (r - dmax).clip(min=0)
    nviol = numpy.count_nonzero(viol)
    rmsd = numpy.sqrt((viol * viol).sum() / weights.sum())
    return nviol, rmsd


def runDynamics(ctx, cq, kernels, collider, indexer,
                coords_buf, masses_buf, radii_buf, nCoords,
                restIndices_buf, restLimits_buf, restWeights_buf, nRest,
                restAmbig_buf, nAmbig, image_idxs_buf, n_image_idxs,
                tRef=1000.0, tStep=0.001, nSteps=1000, fConstR=1.0, fConstD=25.0,
                sConst=0.01, ambigExp=4, beta=10.0, tTaken=0.0,
                printInterval=10000, tot0=20.458):

  if nCoords < 2:
    raise NucCythonError('Too few coodinates')

  tStep0 = tStep * tot0
  beta /= tot0

  (masses, _) = cl.enqueue_map_buffer(
    cq, masses_buf, cl.map_flags.READ,
    0, nCoords, fp_type, is_blocking=False
  )
  (radii, _) = cl.enqueue_map_buffer(
    cq, radii_buf, cl.map_flags.READ,
    0, nCoords, fp_type, is_blocking=False
  )
  (restLimits, _) = cl.enqueue_map_buffer(
    cq, restLimits_buf, cl.map_flags.READ,
    0, (nRest, 2), fp_type, is_blocking=False
  )
  (restIndices, _) = cl.enqueue_map_buffer(
    cq, restIndices_buf, cl.map_flags.READ,
    0, (nRest, 2), dtype('int32'), is_blocking=False
  )
  (restWeights, _) = cl.enqueue_map_buffer(
    cq, restWeights_buf, cl.map_flags.READ,
    0, (nRest,), fp_type, is_blocking=False
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

  image_forces_buf = cl.Buffer(
      ctx, cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.HOST_READ_ONLY |
      cl.mem_flags.READ_WRITE,
      n_image_idxs * 4 * dtype('double').itemsize
  ) if n_image_idxs else None
  image_coords_buf = cl.Buffer(
      ctx, cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.READ_WRITE,
      n_image_idxs * 4 * dtype('double').itemsize
  ) if n_image_idxs else None

  cl.enqueue_fill_buffer(
    cq, accel_buf, numpy.zeros(1, dtype=fp_type),
    0, nCoords * 4 * fp_type.itemsize
  )
  (veloc, _) = cl.enqueue_map_buffer(
    cq, veloc_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
    0, (nCoords, 4), fp_type, is_blocking=False
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
    cq, forces_buf, numpy.zeros(1, dtype=fp_type),
    0, nCoords * 4 * fp_type.itemsize
  )
  e = kernels['getRepulsiveForce'](
    cq, (roundUp(nRep[0], 64),), None,
    repList_buf, forces_buf, coords_buf, radii_buf, masses_buf, fConstR, nRep[0],
    wait_for=[zero_forces]
  )
  e = kernels['getRestraintForce'](
      cq, (roundUp(nAmbig-1, 64),), None,
      restIndices_buf, restLimits_buf, restWeights_buf, restAmbig_buf,
      coords_buf, forces_buf, fConstD, 0.5, ambigExp, nAmbig-1,
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
    if n_image_idxs:
        e_coord_gather = indexer.gather(
            cq, n_image_idxs, coords_buf, image_idxs_buf, image_coords_buf,
            wait_for=[e]
        )
    (veloc, _) = cl.enqueue_map_buffer(
      cq, veloc_buf, cl.map_flags.READ,
      0, (nCoords, 4), fp_type, wait_for=[e], is_blocking=False
    )
    if n_image_idxs:
        (image_coords, _) = cl.enqueue_map_buffer(
            cq, image_coords_buf, cl.map_flags.READ | cl.map_flags.WRITE,
            0, (n_image_idxs, 4), dtype('float64'),
            wait_for=[e_coord_gather], is_blocking=False
        )
    zero_forces = cl.enqueue_fill_buffer(
      cq, forces_buf, numpy.zeros(1, dtype=fp_type),
      0, nCoords * 4 * fp_type.itemsize, wait_for=[e]
    )
    e_rest_force = kernels['getRestraintForce'](
      cq, (roundUp(nAmbig-1, 64),), None,
      restIndices_buf, restLimits_buf, restWeights_buf, restAmbig_buf,
      coords_buf, forces_buf, fConstD, 0.5, ambigExp, nAmbig-1,
      wait_for=[zero_forces]
    )
    if n_image_idxs:
        e_force_gather = indexer.gather(
            cq, n_image_idxs, forces_buf, image_idxs_buf, image_forces_buf,
            wait_for=[e_rest_force]
        )
    e_rep_force = kernels['getRepulsiveForce'](
      cq, (roundUp(nRep[0], 64),), None,
      repList_buf, forces_buf, coords_buf, radii_buf, masses_buf, fConstR, nRep[0],
      wait_for=[zero_forces] + ([e_force_gather] if n_image_idxs else [])
    )
    if n_image_idxs:
        (image_forces, _) = cl.enqueue_map_buffer(
            cq, image_forces_buf, cl.map_flags.READ,
            0, (n_image_idxs, 4), dtype('float64'),
            wait_for=[e_force_gather], is_blocking=False
        )
        cl.wait_for_events([cl.enqueue_barrier(cq)])

        scaling = 1.0 + sConst * (
            divergence(image_coords, image_forces) / (fConstD * radii.mean() ** 2)
        )
        image_coords *= scaling
        del image_forces, image_coords

    r = beta * (tRef / max(getTemp(masses, veloc[:, :3]), 0.001) - 1.0)
    del veloc

    if n_image_idxs:
        indexer.scatter(
            cq, n_image_idxs, image_coords_buf, image_idxs_buf, coords_buf,
        )
    e = kernels['updateVelocity'](
      cq, (roundUp(nCoords, 64),), None,
      veloc_buf, masses_buf, forces_buf, accel_buf, tStep0, r, nCoords,
    )

    (veloc, _) = cl.enqueue_map_buffer(
      cq, veloc_buf, cl.map_flags.READ,
      0, (nCoords, 4), fp_type, wait_for=[e], is_blocking=False
    )
    cl.wait_for_events([cl.enqueue_barrier(cq)])

    if (printInterval > 0) and step % printInterval == 0:
      temp = getTemp(masses, veloc[:, :3])
      (coords, _) = cl.enqueue_map_buffer(
        cq, coords_buf, cl.map_flags.READ,
        0, (nCoords, 4), fp_type, wait_for=[e], is_blocking=False
      )
      if n_image_idxs:
        (image_coords, _) = cl.enqueue_map_buffer(
            cq, image_coords_buf, cl.map_flags.WRITE,
            0, (n_image_idxs, 4), dtype('float64'),
            wait_for=[e_coord_gather], is_blocking=False
        )
      cl.wait_for_events([cl.enqueue_barrier(cq)])
      image_scale = std(image_coords[:, :3]) if n_image_idxs else float('nan')
      if n_image_idxs:
        del image_coords
      scale = std(coords[:, :3])
      nViol, rmsd = getStats(restIndices, restLimits, restWeights, coords[:, :3])

      del coords
      data = (temp, rmsd, nViol, nRep[0], scale, image_scale)
      print('temp:%7.2lf  rmsd:%7.2lf  nViol:%5d  nRep:%5d scale: %7.2f image-scale: %7.2f' % data)

    tTaken += tStep

  return tTaken
