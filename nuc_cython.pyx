from libc.math cimport exp, abs, sqrt, ceil
from numpy cimport ndarray, double_t, int_t, dtype
import numpy, time

BOLTZMANN_K = 0.0019872041

cdef packed struct Restraint_t:
  int indices[2]
  double dists[2]
  int ambiguity

Restraint = dtype([('indices', 'int32', 2), ('dists', 'float64', 2), ('ambiguity', 'int32')])

#ctypedef int_t   int
#ctypedef double_t double

class NucCythonError(Exception):

  def __init__(self, err):

    Exception.__init__(self, err)


cdef int getRepulsionList(ndarray[int,   ndim=2] repList,
                          ndarray[double, ndim=2] coords,
                          double repDist,
                          ndarray[double, ndim=1] radii):

  cdef int i, j
  cdef int n = 0
  cdef double dx, dy, dz, d2
  cdef double distLim
  cdef double distLim2

  for i in range(len(coords)-2):
    for j in range(i+2, len(coords)):
      distLim = repDist + radii[i] + radii[j]
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


cdef double getTemp(ndarray[double, ndim=1] masses,
                    ndarray[double, ndim=2] veloc,
                    int nCoords):
  cdef int i
  cdef double kin = 0.0

  for i in range(nCoords):
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


cdef void updateMotion(ndarray[double, ndim=1] masses,
                       ndarray[double, ndim=2] forces,
                       ndarray[double, ndim=2] accel,
                       ndarray[double, ndim=2] veloc,
                       ndarray[double, ndim=2] coords,
                       int nCoords, double tRef,
                       double tStep, double beta):

  cdef int i
  cdef double r, rtStep, temp

  rtStep = 0.5 * tStep * tStep
  temp = getTemp(masses, veloc, nCoords)
  temp = max(temp, 0.001)
  r = beta * (tRef/temp-1.0)

  for i in range(nCoords):

    accel[i,0] = forces[i,0] / masses[i] + r * veloc[i,0]
    accel[i,1] = forces[i,1] / masses[i] + r * veloc[i,1]
    accel[i,2] = forces[i,2] / masses[i] + r * veloc[i,2]

    coords[i,0] += tStep * veloc[i,0] + rtStep * accel[i,0]
    coords[i,1] += tStep * veloc[i,1] + rtStep * accel[i,1]
    coords[i,2] += tStep * veloc[i,2] + rtStep * accel[i,2]

    veloc[i,0] += tStep * accel[i,0]
    veloc[i,1] += tStep * accel[i,1]
    veloc[i,2] += tStep * accel[i,2]


cdef void updateVelocity(ndarray[double, ndim=1] masses,
                         ndarray[double, ndim=2] forces,
                         ndarray[double, ndim=2] accel,
                         ndarray[double, ndim=2] veloc,
                         int nCoords, double tRef,
                         double tStep, double beta):

  cdef int i
  cdef double r, temp

  temp = getTemp(masses, veloc, nCoords)
  #avoid division by 0 temperature
  temp = max(temp, 0.001)
  r = beta * (tRef/temp-1.0)

  for i in range(nCoords):
    veloc[i,0] += 0.5 * tStep * (forces[i,0] / masses[i] + r * veloc[i,0] - accel[i,0])
    veloc[i,1] += 0.5 * tStep * (forces[i,1] / masses[i] + r * veloc[i,1] - accel[i,1])
    veloc[i,2] += 0.5 * tStep * (forces[i,2] / masses[i] + r * veloc[i,2] - accel[i,2])


cdef double getRepulsiveForce(ndarray[int,   ndim=2] repList,
                              ndarray[double, ndim=2] forces,
                              ndarray[double, ndim=2] coords,
                              int nRep, double fConst,
                              ndarray[double, ndim=1] radii):

  cdef int i, j, k
  cdef double dx, dy, dz, d2, dr, rjk
  cdef double force = 0
  cdef double repDist2

  if fConst == 0:
    return force

  for i from 0 <= i < nRep:
    j = repList[i,0]
    k = repList[i,1]
    repDist = radii[j] + radii[k]
    repDist2 = repDist * repDist


    dx = coords[k,0] - coords[j,0]
    if abs(dx) > repDist:
      continue

    dy = coords[k,1] - coords[j,1]
    if abs(dy) > repDist:
      continue

    dz = coords[k,2] - coords[j,2]
    if abs(dz) > repDist:
      continue

    d2 = dx*dx + dy*dy + dz*dz
    if d2 > repDist2:
      continue

    dr = repDist2 - d2
    #energy contribution
    force += fConst * dr * dr
    rjk = 4 * fConst * dr

    dx *= rjk
    dy *= rjk
    dz *= rjk

    #force contributions
    forces[j,0] -= dx
    forces[k,0] += dx

    forces[j,1] -= dy
    forces[k,1] += dy

    forces[j,2] -= dz
    forces[k,2] += dz

  return force


cdef double getRestraintForce(ndarray[double, ndim=2] forces,
                              ndarray[double, ndim=2] coords,
                              ndarray[int,   ndim=2] restIndices,
                              ndarray[double, ndim=2] restLimits,
                              ndarray[int, ndim=1] restAmbig,
                              int nRest, double fConst, double exponent=2.0,
                              double distSwitch=0.5, double asymptote=1.0):

  cdef int i, j, k, n, nAmbig
  cdef double a, b, da, d, dmin, dmax, dx, dy, dz
  cdef double r, r2, s2, rjk, ujk, force = 0, t

  b = asymptote*distSwitch*distSwitch - exponent*distSwitch*distSwitch*distSwitch
  a = distSwitch*distSwitch - asymptote*distSwitch - b/ distSwitch

  i = 0

  while i < nRest:


    nAmbig = restAmbig[i]
    r2 = 0.0

    if nAmbig == 1:
      j = restIndices[i,0]
      k = restIndices[i,1]

      if j != k:
        dx = coords[j,0] - coords[k,0]
        dy = coords[j,1] - coords[k,1]
        dz = coords[j,2] - coords[k,2]
        r2 = dx*dx + dy*dy + dz*dz

    else:
      for n in range(nAmbig):
        j = restIndices[i+n,0]
        k = restIndices[i+n,1]

        if j != k:
          dx = coords[j,0] - coords[k,0]
          dy = coords[j,1] - coords[k,1]
          dz = coords[j,2] - coords[k,2]
          r = dx*dx + dy*dy + dz*dz
          r2 += 1.0 / (r * r)

      if r2 > 0:
        r2 = 1.0 / sqrt(r2)

    if r2 <= 0:
      i += nAmbig
      continue

    dmin = restLimits[i,0]
    dmax = restLimits[i,1]

    da = dmax + distSwitch

    if r2 < dmin*dmin:
      r2 = max(r2, 1e-8)
      r = sqrt(r2)
      d = dmin - r
      ujk = fConst * d * d
      rjk = fConst * 2 * d

    elif r2 > dmax*dmax:
      r = sqrt(r2)
      d = r - dmax

      if r <= da:
        ujk = fConst * d * d
        rjk = - fConst * 2 * d

      else:
        ujk = fConst * (a + asymptote*d + b/d)
        rjk = - fConst * (asymptote - b/(d*d))

    else:
      ujk = rjk = 0
      r = 1.0

    force += ujk

    if nAmbig == 1:
      j = restIndices[i,0]
      k = restIndices[i,1]

      if j == k:
        i += nAmbig
        continue

      t = rjk / r
      dx = coords[j,0] - coords[k,0]
      dy = coords[j,1] - coords[k,1]
      dz = coords[j,2] - coords[k,2]

      dx *= t
      dy *= t
      dz *= t

      forces[j,0] += dx
      forces[j,1] += dy
      forces[j,2] += dz

      forces[k,0] -= dx
      forces[k,1] -= dy
      forces[k,2] -= dz

    else:

      for n in range(nAmbig):
        j = restIndices[i+n,0]
        k = restIndices[i+n,1]

        if j == k:
          continue

        dx = coords[j,0] - coords[k,0]
        dy = coords[j,1] - coords[k,1]
        dz = coords[j,2] - coords[k,2]

        s2 = dx*dx + dy*dy + dz*dz
        t = rjk * r2 * r2 * r / (s2 * s2 * s2)

        dx *= t
        dy *= t
        dz *= t

        forces[j,0] += dx
        forces[k,0] -= dx
        forces[j,1] += dy
        forces[k,1] -= dy
        forces[j,2] += dz
        forces[k,2] -= dz

    i += nAmbig

  return force


def getSupportedPairs(ndarray[int, ndim=2] positions,
                     int threshold=2000000, int posErr=100):

  cdef int i, j, n = len(positions)
  cdef int pA, pB, pC, pD
  cdef ndarray[int, ndim=1] supported = numpy.zeros(n, numpy.int32)

  for i in range(n):
    pA = positions[i,0]
    pB = positions[i,1]

    for j in range(n):
      if j == i:
        continue

      pC = positions[j,0]
      pD = positions[j,1]

      if (posErr < abs(pC-pA) < threshold) and (posErr < abs(pD-pB) < threshold):
        supported[i] = 1
        break

      elif (posErr < abs(pD-pA) < threshold) and (posErr < abs(pC-pB) < threshold):
        supported[i] = 1
        break

  indices = supported.nonzero()[0]

  return indices


def getInterpolatedCoords(ndarray[double, ndim=2] coords,
                          ndarray[int, ndim=1] positions,
                          ndarray[int, ndim=1] prevPositions):
  """
  Interpolate x,y,z particle positions for an array of seq positions to a new
  seq positions e.g. for a change in binned resolution.
  """

  cdef int n = len(positions), m = len(prevPositions)
  cdef ndarray[double, ndim=2] newCoords = numpy.empty((len(positions), 3), float)

  for i in range(n):
    #find closest old positions for coordinate interpolation
    p1 = 0
    dMin = positions[i]-prevPositions[0]

    for j in range(1,m):
      d = positions[i]-prevPositions[j]

      if abs(d) < abs(dMin):
        p1 = j
        dMin = d #closest pos

      elif abs(d) > abs(dMin): # Seq positions were in order
        break

    if dMin == 0: #new position coincides with an old position
      p2 = p1

    elif dMin > 0: #new pos is above p1
      p2 = min(p1+1, m-1)

    else: #new pos is below p1
      p2 = p1
      p1 = max(0, p1-1)
      dMin = positions[i] - prevPositions[p1]

    #calculate coordinates
    if prevPositions[p2] == prevPositions[p1]:
      newCoords[i, 0] = coords[p1, 0]
      newCoords[i, 1] = coords[p1, 1]
      newCoords[i, 2] = coords[p1, 2]

    else: #interpolate
      f = <float>dMin/<float>(prevPositions[p2]-prevPositions[p1])
      g = 1.0 - f

      newCoords[i, 0] = g * coords[p1, 0] + f * coords[p2, 0]
      newCoords[i, 1] = g * coords[p1, 1] + f * coords[p2, 1]
      newCoords[i, 2] = g * coords[p1, 2] + f * coords[p2, 2]

  return newCoords


def runDynamics(ndarray[double, ndim=2] coords,
                ndarray[double, ndim=1] masses,
                ndarray[double, ndim=1] radii,
                ndarray[int, ndim=2] restIndices,
                ndarray[double, ndim=2] restLimits,
                ndarray[int, ndim=1] restAmbig,
                double tRef=1000.0, double tStep=0.001, int nSteps=1000,
                double fConstR=1.0, double fConstD=25.0, double beta=10.0,
                double repDist=1.5, # A heuristic to limit the repulsion list
                double tTaken=0.0, int printInterval=10000,
                double tot0=20.458):

  cdef int nRest = len(restIndices)
  cdef int nCoords = len(coords)

  if nCoords < 2:
    raise NucCythonError('Too few coodinates')

  indices = set(restIndices.ravel())
  if min(indices) < 0:
    raise NucCythonError('Restraint index negative')

  if max(indices) >= nCoords:
    data = (max(indices), nCoords)
    raise NucCythonError('Restraint index "%d" out of bounds (> %d)' % data)

  if nCoords != len(masses):
    raise NucCythonError('Masses list size does not match coordinates')

  if nRest != len(restLimits):
    raise NucCythonError('Number of restraint index pairs does not match number of restraint limits')

  if len(restAmbig) != nRest:
    raise NucCythonError('Size of ambiguity list does not match number of restraints')

  cdef int i, j, n, step, nViol, nRep = 0

  cdef double d2, dx, dy, dz, ek, rmsd, tStep0, temp, fDist, fRep
  cdef double deltaLim = 0.25 * repDist * repDist
  cdef double Langevin_gamma

  tStep0 = tStep * tot0
  beta /= tot0

  cdef ndarray[double, ndim=2] veloc = numpy.random.normal(0.0, 1.0, (nCoords, 3))
  veloc *= sqrt(tRef / getTemp(masses, veloc, nCoords))

  cdef ndarray[int, ndim=2] repList = numpy.empty((0, 2), numpy.int32)
  cdef ndarray[double, ndim=2] coordsPrev = numpy.array(coords)
  cdef ndarray[double, ndim=2] accel = numpy.zeros((nCoords, 3))
  cdef ndarray[double, ndim=2] forces = numpy.zeros((nCoords, 3))

  cdef double t0 = time.time()

  nRep = getRepulsionList(repList, coords, repDist, radii)
  # Allocate with some padding
  repList = numpy.resize(repList, (int(nRep * 1.2), 2))
  nRep = getRepulsionList(repList, coords, repDist, radii)

  fRep = getRepulsiveForce(repList, forces, coords, nRep,  fConstR, radii)
  fDist = getRestraintForce(forces, coords, restIndices, restLimits, restAmbig, nRest, fConstD)

  for step in range(nSteps):
    for i in range(nCoords):
      dx = coords[i,0] - coordsPrev[i,0]
      dy = coords[i,1] - coordsPrev[i,1]
      dz = coords[i,2] - coordsPrev[i,2]
      if dx*dx + dy*dy + dz*dz > deltaLim:
        nRep = getRepulsionList(repList, coords, repDist, radii)
        if nRep > len(repList):
          repList = numpy.resize(repList, (int(nRep * 1.2), 2))
          nRep = getRepulsionList(repList, coords, repDist, radii)
        elif nRep < (len(repList) // 2):
          repList = numpy.resize(repList, (int(nRep * 1.2), 2))

        for i in range(nCoords):
          coordsPrev[i,0] = coords[i,0]
          coordsPrev[i,1] = coords[i,1]
          coordsPrev[i,2] = coords[i,2]
        break # Already re-calculated, no need to check more

    updateMotion(masses, forces, accel, veloc, coords, nCoords, tRef, tStep0, beta)

    for i in range(nCoords):
      forces[i,0] = 0.0
      forces[i,1] = 0.0
      forces[i,2] = 0.0

    fRep  = getRepulsiveForce(repList, forces, coords, nRep, fConstR, radii)
    fDist = getRestraintForce(forces, coords, restIndices, restLimits, restAmbig, nRest, fConstD)

    updateVelocity(masses, forces, accel, veloc, nCoords, tRef, tStep0,  beta)

    if (printInterval > 0) and step % printInterval == 0:
      temp = getTemp(masses, veloc, nCoords)
      nViol, rmsd = getStats(restIndices, restLimits, coords, nRest)

      data = (temp, fRep, fDist, rmsd, nViol, nRep)
      print('temp:%7.2lf  fRep:%7.2lf  fDist:%7.2lf  rmsd:%7.2lf  nViol:%5d  nRep:%5d' % data)

    tTaken += tStep

  return tTaken


def calc_restraints(contact_dict, pos_dict, int particle_size=10000,
                    float scale=1.0, float exponent=-0.33,
                    float lower=0.8, float upper=1.2,
                    int min_count=1):
  """
  Function to convert single-cell contact data into distance restraints
  for structure calculations.
  """

  cdef int i, j, k, a, b, nc, n, na, nb
  cdef double dist
  cdef ndarray[long, ndim=2] contacts # Contact matrix (4:(posA, posB, nObs, ambigGrp), nContacts)
  cdef ndarray[Restraint_t, ndim=1] restraints  # Distance restraints (nRestraints,)
  cdef ndarray[int, ndim=2] bin_matrix     # Temp array for binned contacts
  cdef ndarray[int, ndim=1] seq_pos_a
  cdef ndarray[int, ndim=1] seq_pos_b
  cdef ndarray[int, ndim=2] limits         # shape: (chromoId, 2:[start, end])

  restraint_dict = {}    # Final restraints for each pair of chromosomes

  chromos = set(pos_dict)

  # Get restraint indices, do binning of contact observations
  for chrA in contact_dict:
    if chrA not in chromos:
      continue

    na = len(pos_dict[chrA])
    seq_pos_a = pos_dict[chrA]
    restraint_dict[chrA] = {}

    for chrB in contact_dict[chrA]:
      if chrB not in chromos:
        continue

      nb = len(pos_dict[chrB])
      seq_pos_b = pos_dict[chrB]

      contacts = contact_dict[chrA][chrB]
      n = len(contacts[0])

      restraints = numpy.empty(n, Restraint)
      bin_matrix = numpy.zeros((na, nb), numpy.int32)

      for i in range(n):

        # Find bin index for chromo A
        for j in range(na):
          if seq_pos_a[j] >= contacts[0,i]:
            break

        else:
          continue

        # Find bin index for chromo B
        for k in range(nb):
          if seq_pos_b[k] >= contacts[1,i]:
            break

        else:
          continue

        bin_matrix[j,k] += contacts[2,i]

      #loop over all binned contacts, and calculate the constraint target distance
      #using a powerlaw function and the number of observations
      k = 0
      for i in range(na):
        for j in range(nb):
          if bin_matrix[i,j] > 0:
            if bin_matrix[i,j] < min_count:
              continue

            dist = scale * bin_matrix[i,j] ** exponent

            restraints[k].indices[0] = i # binA
            restraints[k].indices[1] = j # binB
            restraints[k].dists[0] = dist * lower # constraint lower bound
            restraints[k].dists[1] = dist * upper # constraint upper bound
            restraints[k].ambiguity = 0 # Use '0' to represent no ambiguity

            k += 1

      restraint_dict[chrA][chrB] = restraints[:k]

  return restraint_dict


def concatenate_restraints(restraint_dict, pos_dict):
  """
  Joins restraints stored in a dict by chromo pairs into long concatenated arrays.
  Indices of restraints relate to concatenated chromo seq pos.
  Add-in all the backbone restraints for sequential particles.
  """

  cdef int i, n, num_restraints, m
  cdef int start_a, start_b

  # Get total max number restraints and final restraint index offset for all chromos
  i = 0
  num_restraints = 0
  chromo_idx_offset = {}

  for chr_a in sorted(pos_dict):
    n = len(pos_dict[chr_a])
    chromo_idx_offset[chr_a] = i
    i += n

  for chr_a in restraint_dict:
    for chr_b in restraint_dict[chr_a]:
      num_restraints += len(restraint_dict[chr_a][chr_b])

  # Loop allocated arrays
  cdef ndarray[int, ndim=1] positions
  cdef ndarray[Restraint_t, ndim=1] restraints

  # Final arrays which will hold identities of restrained particle pairs
  # and the restraint distances for each
  cdef ndarray[int, ndim=2] particle_indices = numpy.empty((num_restraints, 2), numpy.int32)
  cdef ndarray[double, ndim=2] distances = numpy.empty((num_restraints, 2), float)
  cdef ndarray[int, ndim=1] ambiguity_groups = numpy.empty(num_restraints, numpy.int32)

  m = 0

  # Add regular restraints for chromo pairs
  for chr_a in restraint_dict:
    start_a = chromo_idx_offset[chr_a] # Offset for chromo A particles

    for chr_b in restraint_dict[chr_a]:
      start_b = chromo_idx_offset[chr_b] # Offset for chromo B particles

      restraints = restraint_dict[chr_a][chr_b]

      for i in range(len(restraints)):
        particle_indices[m,0] = restraints[i].indices[0] + start_a
        particle_indices[m,1] = restraints[i].indices[1] + start_b

        distances[m,0] = restraints[i].dists[0] # lower
        distances[m,1] = restraints[i].dists[1] # upper

        ambiguity_groups[m] = restraints[i].ambiguity

        m += 1

  return particle_indices, distances, ambiguity_groups


def calc_ambiguity_strides(ndarray[int, ndim=1] groups):
  """
  Convert (sorted) ambiguity groups to ambiguity strides suitable for
  annealing calculations.
  """
  cdef int i = 0, count = 1, current_group = 0
  cdef ndarray[int, ndim=1] strides = numpy.empty(len(groups), 'int32')

  for i in range(len(groups)):
    if groups[i] == current_group == 0:
      strides[i] = 1
    elif groups[i] == current_group:
      count += 1
    else:
      strides[i-count:i] = count
      current_group = groups[i]
      count = 1
  strides[i+1-count:i+1] = count

  return strides
