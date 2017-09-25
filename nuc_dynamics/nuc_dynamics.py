from pathlib import Path
from numpy import dtype
import pyopencl as cl

from .run import runDynamics

fp_type = dtype('float64')

Restraint = dtype([('indices', 'int32', 2), ('dists', fp_type, 2),
                   ('ambiguity', 'int32'), ('weight', fp_type)])
Contact = dtype([('pos', 'uint32', 2), ('ambiguity', 'int32')])


def load_points(path):
    if path.suffix == '.txt':
        from numpy import loadtxt
        return loadtxt(str(path))
    if path.suffix == '.csv':
        from numpy import loadtxt
        return loadtxt(str(path), delimiter=',')
    elif path.suffix == '.pickle':
        with path.open('rb') as f:
            return load(f)
    else:
        raise ValueError("File type '{}' not recognized (need '.csv' or '.pickle')"
                         .format(path.suffix))


def load_ncc_file(file_path):
  """Load chromosome and contact data from NCC format file, as output from NucProcess"""

  from numpy import array

  if file_path.endswith('.gz'):
    import gzip
    f_open = gzip.open
  else:
    f_open = open

  contact_dict = {}

  with f_open(file_path) as file_obj:
    for line in file_obj:
      if line.startswith('#'):
          continue
      chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, ambig_group, pair_id, swap_pair = line.split()

      pos_a = int(f_start_a if strand_a == '+' else f_end_a)
      pos_b = int(f_start_b if strand_b == '+' else f_end_b)

      if chr_a > chr_b:
        chr_a, chr_b = chr_b, chr_a
        pos_a, pos_b = pos_b, pos_a

      if chr_a not in contact_dict:
        contact_dict[chr_a] = {}

      if chr_b not in contact_dict[chr_a]:
        contact_dict[chr_a][chr_b] = []

      contact_dict[chr_a][chr_b].append(((pos_a, pos_b), int(ambig_group)))

  for chr_a in contact_dict:
    for chr_b in contact_dict[chr_a]:
      contact_dict[chr_a][chr_b] = array(contact_dict[chr_a][chr_b], dtype=Contact)
  return contact_dict


def calc_limits(contact_dict):
  chromo_limits = {}

  for chrs, contacts in flatten_dict(contact_dict).items():
    if len(contacts) < 1:
      continue

    for chr, seq_pos in zip(chrs, contacts['pos'].T):
      min_pos = seq_pos.min()
      max_pos = seq_pos.max()

      prev_min, prev_max = chromo_limits.get(chr, (min_pos, max_pos))
      chromo_limits[chr] = [min(prev_min, min_pos), max(prev_max, max_pos)]
  return chromo_limits


def load_image_coords(file_path):
  from pickle import load

  with file_path.open("rb") as f:
    return load(f)


def export_nuc_coords(file_path, coords_dict, seq_pos_dict, restraint_dict, calc_args):
  import h5py

  with h5py.File(file_path, "w") as f:
    structure = f.create_group("structures/0")
    coords = structure.create_group("coords")
    for chromosome, data in coords_dict.items():
      coords.create_dataset(chromosome, data=data)
    seq_pos = structure.create_group("particles")
    for chromosome, data in seq_pos_dict.items():
      seq_pos.create_group(chromosome).create_dataset('positions', data=data)
    calculation = structure.create_group('calculation')
    for arg, value in calc_args.items():
      if value is None:
        value = 'None'
      elif isinstance(value, Path):
        value = str(value)
      elif isinstance(value, tuple):
        value = list(value)
      calculation.attrs[arg] = value
    restraints = structure.create_group('restraints')
    for (chr_a, chr_b), data in flatten_dict(restraint_dict).items():
      group = restraints.require_group(chr_a)
      group.create_dataset(chr_b, data=data)


def between(x, a, b):
  return (a < x) & (x < b)


def remove_isolated_contacts(contact_dict, threshold=int(2e6), pos_error=100, ignore=()):
  """
  Select only contacts which are within a given sequence separation of another
  contact, for the same chromosome pair
  """
  from numpy import array, zeros
  from collections import defaultdict
  new_contacts = defaultdict(dict)

  for (chromoA, chromoB), contacts in flatten_dict(contact_dict).items():
    positions = contacts['pos'].astype('int32')
    if chromoA in ignore or chromoB in ignore:
        new_contacts[chromoA][chromoB] = contacts
        continue
    if len(positions) == 0: # Sometimes empty e.g. for MT, Y chromos
      continue
    pos_a = positions[:, 0][None, ...]
    pos_b = positions[:, 1][None, ...]

    supported = zeros(len(positions), dtype='bool')
    supported |= (between(abs(pos_a - pos_a.T), pos_error, threshold).any(axis=0) &
                  between(abs(pos_b - pos_b.T), pos_error, threshold).any(axis=0))
    supported |= (between(abs(pos_a - pos_b.T), pos_error, threshold).any(axis=0) &
                  between(abs(pos_b - pos_a.T), pos_error, threshold).any(axis=0))

    if not supported.any():
        continue
    new_contacts[chromoA][chromoB] = contacts[supported]
  return dict(new_contacts)


def remove_violated_contacts(contact_dict, coords_dict, particle_seq_pos, threshold=5.0):
  """
  Remove contacts whith structure distances that exceed a given threshold
  """
  from numpy import int32, sqrt, array
  from collections import defaultdict
  new_contacts = defaultdict(dict)

  for chr_a in contact_dict:
    if chr_a not in coords_dict:
      continue
    for chr_b, contacts in contact_dict[chr_a].items():
      if chr_b not in coords_dict:
        continue

      contact_pos_a = contacts['pos'][:, 0].astype(int32)
      contact_pos_b = contacts['pos'][:, 1].astype(int32)

      coords_a = coords_dict[chr_a]
      coords_b = coords_dict[chr_b]

      struc_dists = []

      for m in range(len(coords_a)):
        coord_data_a = get_interpolated_coords(coords_a[m], contact_pos_a, particle_seq_pos[chr_a])
        coord_data_b = get_interpolated_coords(coords_b[m], contact_pos_b, particle_seq_pos[chr_b])

        deltas = coord_data_a - coord_data_b
        dists = sqrt((deltas*deltas).sum(axis=1))
        struc_dists.append(dists)

      # Average over all conformational models
      struc_dists = array(struc_dists).T.mean(axis=1)

      # Select contacts with distances below distance threshold
      indices = (struc_dists < threshold).nonzero()[0]
      new_contacts[chr_a][chr_b] = contacts[indices]

  return dict(new_contacts)


def get_random_coords(shape, radius=10.0):
  """
  Get random, uniformly sampled coorinate positions, restricted to
  a nD-sphere of given radius
  """

  from numpy import random
  from numpy.linalg import norm

  u = random.uniform(size=shape[:-1])
  x = random.normal(size=shape)
  scaling = (radius * u ** (1/shape[-1])) / norm(x, axis=-1)
  return scaling[..., None] * x


def get_interpolated_coords(coords, pos, prev_pos):
  from numpy import interp, apply_along_axis
  from functools import partial

  *size, length, ndim = coords.shape
  coords = coords.reshape((-1, length, ndim))
  coords = apply_along_axis(partial(interp, pos, prev_pos), 1, coords)
  return coords.reshape(size + [len(pos), ndim])


def unpack_chromo_coords(coords, chromosomes, seq_pos_dict):
  """
  Exctract coords for multiple chromosomes stored in a single array into
  a dictionary, keyed by chromosome name. The chromosomes argument is required
  to get the correct array storage order.
  """

  coords_dict = {}

  j = 0
  for i, chromo in enumerate(chromosomes):
    span = len(seq_pos_dict[chromo])
    coords_dict[chromo] = coords[:,j:j+span] # all models, slice
    j += span

  return coords_dict


def calc_bins(chromo_limits, particle_size):
  from numpy import arange
  from math import ceil

  bins = {}
  for chr, (_, end) in chromo_limits.items():
    end = (end // particle_size + bool(end % particle_size)) * particle_size
    # TODO (kjw53): Original uses ceil, then adds additional particle. Why?
    end += particle_size

    bins[chr] = arange(0, end, particle_size, dtype='int32')
  return bins


def calc_ambiguity_offsets(groups):
  """
  Convert (sorted) ambiguity groups to group-offsets for
  annealing calculations.
  """
  from numpy import flatnonzero, zeros

  group_starts = zeros(len(groups) + 1, dtype='bool')
  group_starts[-1] = group_starts[0] = 1
  group_starts[:-1] |= groups == 0
  group_starts[1:-1] |= groups[1:] != groups[:-1]
  return flatnonzero(group_starts).astype('int32')


def backbone_restraints(seq_pos, particle_size, scale=1.0, lower=0.1, upper=1.1, weight=1.0):
  from numpy import empty, arange, array

  restraints = empty(len(seq_pos) - 1, dtype=Restraint)
  offsets = array([0, 1], dtype='int')
  restraints['indices'] = arange(len(restraints))[:, None] + offsets

  # Normally 1.0 for regular sized particles
  bounds = array([lower, upper], dtype=fp_type) * scale
  restraints['dists'] = ((seq_pos[1:] - seq_pos[:-1]) / particle_size)[:, None] * bounds
  restraints['ambiguity'] = 0 # Use '0' to represent no ambiguity
  restraints['weight'] = weight

  return restraints


def flatten_dict(d):
  r = {}
  for key, value in d.items():
    if isinstance(value, dict):
      r.update({(key,) + k: v for k, v in flatten_dict(value).items()})
    else:
      r[(key,)] = value
  return r


def tree():
    from collections import defaultdict
    def tree_():
        return defaultdict(tree_)
    return tree_()


def unflatten_dict(d):
    r = tree()

    for ks, v in d.items():
        d = r
        for k in ks[:-1]:
            d = d[k]
        d[ks[-1]] = v
    return r


def concatenate_restraints(restraint_dict, pos_dict):
  from itertools import accumulate, chain
  from functools import partial
  from numpy import concatenate, array, empty
  import operator as op

  chromosomes = sorted(pos_dict)
  chromosome_offsets = dict(zip(chromosomes, accumulate(chain(
    [0], map(len, map(partial(op.getitem, pos_dict), chromosomes)))
  )))
  flat_restraints = sorted(flatten_dict(restraint_dict).items())

  r = concatenate(list(map(op.itemgetter(1), flat_restraints)))
  r['indices'] = concatenate([
    restraints['indices'] + array([[chromosome_offsets[c] for c in chrs]])
    for chrs, restraints in flat_restraints
  ])

  return r


def calc_restraints(contact_dict, pos_dict,
                    scale=1.0, lower=0.8, upper=1.2, weight=1.0):
  from numpy import empty, array, searchsorted
  from collections import defaultdict, Counter

  dists = array([[lower, upper]]) * scale

  r = defaultdict(dict)
  for (chr_a, chr_b), contacts in flatten_dict(contact_dict).items():
    if len(contacts) < 1:
        continue
    r[chr_a][chr_b] = empty(len(contacts), dtype=Restraint)

    idxs_a = searchsorted(pos_dict[chr_a], contacts['pos'][:, 0])
    idxs_b = searchsorted(pos_dict[chr_b], contacts['pos'][:, 1])
    r[chr_a][chr_b]['indices'] = array([idxs_a, idxs_b]).T
    r[chr_a][chr_b]['ambiguity'] = contacts['ambiguity']
    r[chr_a][chr_b]['dists'] = dists
    r[chr_a][chr_b]['weight'] = weight
  return dict(r)


def bin_restraints(restraints):
  from numpy import unique, bincount, empty, concatenate, sort, copy
  restraints = copy(restraints)
  restraints['indices'] = sort(restraints['indices'], axis=1)

  # Ambiguity group 0 is unique restraints, so they can all be binned
  _, inverse, counts = unique(
    restraints['ambiguity'], return_inverse=True, return_counts=True
  )
  restraints['ambiguity'][counts[inverse] == 1] = 0

  # Bin within ambiguity groups
  uniques, idxs = unique(
    restraints[['ambiguity', 'indices', 'dists']], return_inverse=True
  )

  binned = empty(len(uniques), dtype=Restraint)
  binned['ambiguity'] = uniques['ambiguity']
  binned['indices'] = uniques['indices']
  binned['dists'] = uniques['dists']
  binned['weight'] = bincount(idxs, weights=restraints['weight'])
  return binned

def merge_dicts(*dicts):
  from functools import partial
  from collections import defaultdict
  from numpy import empty, concatenate

  new = defaultdict(list)
  for d in dicts:
    for k, v in flatten_dict(d).items():
      new[k].append(v)
  new = {k: concatenate(v) for k, v in new.items()}
  return unflatten_dict(new)


def concatenate_into(data, out, axis=0):
  from itertools import accumulate

  sizes = [d.shape[axis] for d in data]
  offsets = list(accumulate([0] + sizes))

  for offset, size, d in zip(offsets, sizes, data):
    indices = [slice(None)] * out.ndim
    indices[axis] = slice(offset, offset + size)
    out[tuple(indices)] = d
  return out


def roundUp(x, base=1):
  return (x // base + bool(x % base)) * base


def lcm(a, b):
  """Lowest common multiple"""
  try:
    from math import gcd
  except ImportError:
    from fractions import gcd

  # GCD is an integer for integer inputs
  return abs(a * b) // gcd(a, b)


def anneal_genome(ctx, cq, kernels, contact_dict, images, particle_size,
                  prev_seq_pos_dict=None, start_coords=None,
                  contact_dist=(0.8, 1.2), backbone_dist=(0.1, 1.1),
                  image_weight=1.0, temp_range=(5000.0, 10.0), temp_steps=500,
                  dynamics_steps=100, time_step=0.001,
                  random_seed=None, random_radius=10.0, num_models=1,
                  ambig_exponent=4, printInterval=100):
    """
    Use chromosome contact data to generate distance restraints and then
    apply a simulated annealing protocul to generate/refine coordinates.
    Starting coordinates may be random of from a previous (e.g. lower
    resolution) stage.
    """

    from numpy import (int32, ones, empty, random, concatenate, stack, argsort, arange,
                       geomspace, linspace, arctan, full, zeros, dtype, ascontiguousarray)
    from math import log, exp, atan, pi
    from functools import partial
    from collision.collision import Collider

    bead_size = particle_size ** (1/3)

    if random_seed is not None:
      random.seed(random_seed)
    particle_size = int32(particle_size)

    chromosomes = sorted(
        set.union(set(contact_dict), *map(set, contact_dict.values())) - images
    )
    images = sorted(images)

    seq_pos_dict = calc_bins(calc_limits(contact_dict), particle_size)
    seq_pos_dict.update({img: arange(start_coords[img].shape[1], dtype='int32') for img in images})

    points = sorted(chromosomes + images)

    restraint_dict = calc_restraints(
        contact_dict, seq_pos_dict, scale=bead_size,
        lower=contact_dist[0], upper=contact_dist[1], weight=1.0
    )

    # Apply image weighting
    for chr_a, chr_b in flatten_dict(restraint_dict):
        if chr_a in images or chr_b in images:
            restraint_dict[chr_a][chr_b]['weight'] *= image_weight

    # Adjust to keep force/particle approximately constant
    dist = 430.0 * (sum(map(len, seq_pos_dict.values())) /
                    sum(map(lambda v: v['weight'].sum(),
                            flatten_dict(restraint_dict).values())))

    restraint_dict = merge_dicts(
      restraint_dict,
      # Backbone restraints
      {chr: {chr: backbone_restraints(
        seq_pos_dict[chr], particle_size, bead_size,
        lower=backbone_dist[0], upper=backbone_dist[1], weight=1.0
      )} for chr in chromosomes}
    )

    coords = start_coords or {}
    for chr in chromosomes:
      if chr not in coords:
        coords[chr] = get_random_coords(
          (num_models, len(seq_pos_dict[chr]), 3), random_radius * bead_size
        )
      elif coords[chr].shape[1] != len(seq_pos_dict[chr]):
        coords[chr] = get_interpolated_coords(
          coords[chr], seq_pos_dict[chr], prev_seq_pos_dict[chr]
        )

    # Equal unit masses and radii for all particles
    masses = {chr: ones(len(pos), dtype=fp_type) for chr, pos in seq_pos_dict.items()}
    masses.update({
        img: full(coords[img].shape[1], float('inf'), dtype=fp_type) for img in images
    })

    radii = {chr: full(len(pos), bead_size, dtype=fp_type)
             for chr, pos in seq_pos_dict.items()}
    # No collisions, so radii don't matter
    radii.update({img: zeros(coords[img].shape[1], dtype=fp_type) for img in images})

    # Concatenate chromosomal data into a single array of particle restraints
    # for structure calculation.
    restraints = bin_restraints(concatenate_restraints(restraint_dict, seq_pos_dict))
    restraint_order = argsort(restraints['ambiguity'])
    restraints = restraints[restraint_order]
    ambiguity = calc_ambiguity_offsets(restraints['ambiguity'])

    n_particles = sum(map(len, seq_pos_dict.values()))
    coord_size = dtype((fp_type, 4)).itemsize
    model_buf_size = roundUp(
      n_particles * coord_size, lcm(coord_size, cq.device.mem_base_addr_align)
    )
    coords_buf_shape = (num_models, model_buf_size // coord_size, 4)

    coords_buf = cl.Buffer(
      ctx, cl.mem_flags.READ_ONLY,
      num_models * model_buf_size
    )
    model_coords_bufs = [
      coords_buf[i * model_buf_size : (i+1) * model_buf_size] for i in range(num_models)
    ]
    masses_buf = cl.Buffer(
      ctx, cl.mem_flags.READ_ONLY, n_particles * fp_type.itemsize
    )
    radii_buf = cl.Buffer(
      ctx, cl.mem_flags.READ_ONLY, num_models * n_particles * fp_type.itemsize
    )
    rest_indices_buf = cl.Buffer(
      ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_ONLY |
      cl.mem_flags.COPY_HOST_PTR, hostbuf=ascontiguousarray(restraints['indices'])
    )
    rest_limits_buf = cl.Buffer(
      ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_ONLY |
      cl.mem_flags.COPY_HOST_PTR, hostbuf=ascontiguousarray(restraints['dists'])
    )
    rest_weights_buf = cl.Buffer(
      ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_ONLY |
      cl.mem_flags.COPY_HOST_PTR, hostbuf=ascontiguousarray(restraints['weight'])
    )
    ambiguity_buf = cl.Buffer(
      ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_ONLY |
      cl.mem_flags.COPY_HOST_PTR, hostbuf=ambiguity
    )

    (coords_map, _) = cl.enqueue_map_buffer(
      cq, coords_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
      0, coords_buf_shape, fp_type, is_blocking=False
    )
    (masses_map, _) = cl.enqueue_map_buffer(
      cq, masses_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
      0, n_particles, fp_type, is_blocking=False
    )
    (radii_map, _) = cl.enqueue_map_buffer(
      cq, radii_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
      0, n_particles, fp_type, is_blocking=False
    )
    cl.wait_for_events([cl.enqueue_barrier(cq)])

    concatenate_into([coords[chr] for chr in points], coords_map[..., :3], axis=1)
    concatenate_into([masses[chr] for chr in points], masses_map)
    concatenate_into([radii[chr] for chr in points], radii_map)

    del coords_map, masses_map, radii_map

    # Setup annealig schedule: setup temps and repulsive terms
    temps = geomspace(temp_range[0], temp_range[1], temp_steps, endpoint=False)
    temps *= bead_size ** 2
    repulses = arctan(linspace(0, 1, temp_steps, endpoint=False) * 20 - 10) / pi / arctan(10) + 0.5
    repulses /= bead_size ** 2

    # Update coordinates in the annealing schedule
    time_taken = 0.0

    collider = Collider(ctx, n_particles, 16, 64, coord_dtype=fp_type)

    for model_coords_buf in model_coords_bufs: # For each repeat calculation
      for temp, repulse in zip(temps, repulses):
        # Update coordinates for this temp
        dt = runDynamics(
          ctx, cq, kernels, collider,
          model_coords_buf, masses_buf, radii_buf, n_particles,
          rest_indices_buf, rest_limits_buf, rest_weights_buf, len(restraints),
          ambiguity_buf, len(ambiguity),
          temp, time_step, dynamics_steps, repulse, dist,
          ambigExp=ambig_exponent, printInterval=printInterval
        )

        time_taken += dt

    (coords_map, _) = cl.enqueue_map_buffer(
      cq, coords_buf, cl.map_flags.READ,
      0, coords_buf_shape, fp_type, is_blocking=True
    )

    # Convert from single coord array to dict keyed by chromosome
    coords_dict = unpack_chromo_coords(
        coords_map[:, :n_particles, :3], points, seq_pos_dict
    )

    return coords_dict, seq_pos_dict, restraint_dict


def hierarchical_annealing(ctx, cq, kernels, start_coords, contacts, images,
                           particle_sizes, num_models=1, **kwargs):
    from numpy import broadcast_to

    # Unspecified coords will be random
    start_coords = {k: broadcast_to(v, (num_models,) + v.shape)
                    for k, v in start_coords.items()}

    # Record partile positions from previous stages
    # so that coordinates can be interpolated to higher resolution
    prev_seq_pos = None

    for stage, particle_size in enumerate(particle_sizes):

        print("Running structure caculation stage %d (%d kb)" % (stage+1, (particle_size/1e3)))

        # Can remove large violations (noise contacts inconsistent with structure)
        # once we have a resonable resolution structure
        if stage == 4:
            remove_violated_contacts(
                contacts, coords_dict, particle_seq_pos, threshold=6.0
            )
        elif stage == 5:
            remove_violated_contacts(
                contacts, coords_dict, particle_seq_pos, threshold=5.0
            )

        coords_dict, particle_seq_pos, restraint_dict = anneal_genome(
            ctx, cq, kernels, contacts, images, particle_size,
            prev_seq_pos, start_coords, num_models=num_models, **kwargs
        )

        # Next stage based on previous stage's 3D coords
        # and thier respective seq. positions
        start_coords = coords_dict
        prev_seq_pos = particle_seq_pos

    return coords_dict, particle_seq_pos, restraint_dict


def compile_kernels(ctx):
  from numpy import int32, uint32

  with (Path(__file__).parent / "kernels.cl").open() as f:
    program = cl.Program(ctx, f.read()).build()
  kernels = ['updateVelocity', 'updateMotion', 'getRepulsiveForce', 'getRestraintForce']
  kernels = {name: getattr(program, name) for name in kernels}
  kernels['updateVelocity'].set_scalar_arg_dtypes(
    [None, None, None, None, fp_type, fp_type, uint32]
  )
  kernels['updateMotion'].set_scalar_arg_dtypes(
    [None, None, None, None, None, fp_type, fp_type, uint32]
  )
  kernels['getRepulsiveForce'].set_scalar_arg_dtypes(
    [None, None, None, None, None, fp_type, uint32]
  )
  kernels['getRestraintForce'].set_scalar_arg_dtypes(
    [None, None, None, None, None, None, fp_type, fp_type, uint32, uint32]
  )
  return kernels


def main(args=None):
    from argparse import ArgumentParser
    from sys import argv
    from numpy import concatenate, array

    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)
    kernels = compile_kernels(ctx)

    parser = ArgumentParser(description="Calculate a structure from a contact file.")
    parser.add_argument("contacts", type=Path, nargs='+',
                        help="The .ncc file to load contacts from")
    parser.add_argument("output", type=Path, help="The .nuc file to save the structure in")
    parser.add_argument("--image", type=str, nargs=2, action='append', default=[],
                        help="The name and file to load coordinates from")
    parser.add_argument("--isolated-threshold", type=float, default=2e6,
                        help="The distance threshold for isolated contacts")
    parser.add_argument("--particle-sizes", type=float, nargs='+',
                        default=[8e6, 4e6, 2e6, 4e5, 2e5, 1e5],
                        help="The resolutions to calculate structures at")
    parser.add_argument("--models", type=int, default=1,
                        help="The number of models to calculate")
    parser.add_argument("--contact-dist", type=float, nargs=2, default=(0.8, 1.2),
                        help="The upper and lower contact restraint distances")
    parser.add_argument("--backbone-dist", type=float, nargs=2, default=(0.1, 1.1),
                        help="The upper and lower backbone restraint distances")
    parser.add_argument("--temp-range", type=float, nargs=2, default=(5000.0, 10.0),
                        help="The range of 'temperatures' to use for the annealing protocol")
    parser.add_argument("--temp-steps", type=int, default=500,
                        help="The number of temperature steps for the annealing protocol")
    parser.add_argument("--dyn-steps", type=int, default=100,
                        help="The number of dynamics steps for the annealing protocol")
    parser.add_argument("--time-step", type=float, default=0.001,
                        help="The time-step for the annealing protocol")
    parser.add_argument("--seed", type=int, default=None,
                        help="The random seed for starting coordinates")
    parser.add_argument("--random-radius", type=float, default=10,
                        help="The radius for random starting coordinates")
    parser.add_argument("--image-weight", type=float, default=1.0,
                        help="The weighting to use for image-based restraints")
    parser.add_argument("--image-scale", type=float, default=1.0,
                        help="The scaling of the image coordinates")
    parser.add_argument("--print-interval", type=int, default=0,
                        help="The number of dynamics steps between stat logs")
    parser.add_argument("--ambiguous-exponent", type=int, default=4,
                        help="The exponent to use for ambiguous distances")

    args = parser.parse_args(argv[1:] if args is None else args)

    # Load image coordinates, center and scale
    image_coords = {name: load_points(Path(path)) for name, path in args.image}
    if image_coords:
        image_mean = concatenate(list(image_coords.values())).mean(axis=0)
        image_coords = {k: (v - image_mean) * args.image_scale
                        for k, v in image_coords.items()}

    contacts = merge_dicts(*map(load_ncc_file, map(str, args.contacts)))
    contacts = remove_isolated_contacts(
        contacts, threshold=args.isolated_threshold, ignore=image_coords
    )

    coords, seq_pos, restraints = hierarchical_annealing(
        ctx, cq, kernels, image_coords, contacts, set(image_coords.keys()),
        args.particle_sizes,
        contact_dist=args.contact_dist, backbone_dist=args.backbone_dist,
        image_weight=args.image_weight,
        # Cautious annealing parameters
        # Don' need to be fixed, but are for simplicity
        temp_range=args.temp_range, temp_steps=args.temp_steps,
        dynamics_steps=args.dyn_steps, time_step=args.time_step,
        # To set up starting coords
        random_seed=args.seed, random_radius=args.random_radius,
        num_models=args.models, printInterval=args.print_interval,
        ambig_exponent=args.ambiguous_exponent
    )

    # Does not deal with sequences of variable length strings
    calc_args = {k: v for k, v in vars(args).items()
                 if k not in {'image', 'contacts'}}
    export_nuc_coords(str(args.output), coords, seq_pos, restraints, calc_args)
    print('Saved structure file to: %s' % str(args.output))

if __name__ == "__main__":
    main()
