from pathlib import Path

from .nuc_cython import (runDynamics, getSupportedPairs, Restraint)

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

      contact_dict[chr_a][chr_b].append((pos_a, pos_b, int(ambig_group)))

  for chr_a in contact_dict:
    for chr_b in contact_dict[chr_a]:
      contact_dict[chr_a][chr_b] = array(contact_dict[chr_a][chr_b]).T
  return contact_dict


def calc_limits(contact_dict):
  chromo_limits = {}

  for chrs, contacts in flatten_dict(contact_dict).items():
    if contacts.shape[1] < 1:
      continue

    for chr, seq_pos in zip(chrs, contacts):
      min_pos = seq_pos.min()
      max_pos = seq_pos.max()

      prev_min, prev_max = chromo_limits.get(chr, (min_pos, max_pos))
      chromo_limits[chr] = [min(prev_min, min_pos), max(prev_max, max_pos)]
  return chromo_limits


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
      calculation.attrs[arg] = value
    restraints = structure.create_group('restraints')
    for (chr_a, chr_b), data in flatten_dict(restraint_dict).items():
      group = restraints.require_group(chr_a)
      group.create_dataset(chr_b, data=data)


def remove_isolated_contacts(contact_dict, threshold=int(2e6)):
  """
  Select only contacts which are within a given sequence separation of another
  contact, for the same chromosome pair
  """
  from numpy import array, int32
  from collections import defaultdict
  new_contacts = defaultdict(dict)

  for chromoA in contact_dict:
    for chromoB, contacts in contact_dict[chromoA].items():
      positions = array(contacts[:2], int32).T

      if len(positions): # Sometimes empty e.g. for MT, Y chromos
        active_idx = getSupportedPairs(positions, int32(threshold))
        new_contacts[chromoA][chromoB] = contacts[:,active_idx]

  return dict(new_contacts)


def remove_violated_contacts(contact_dict, coords_dict, particle_seq_pos, particle_size, threshold=5.0):
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

      contact_pos_a = contacts[0].astype(int32)
      contact_pos_b = contacts[1].astype(int32)

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
      new_contacts[chr_a][chr_b] = contacts[:,indices]

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

  return apply_along_axis(partial(interp, pos, prev_pos), 0, coords)


def unpack_chromo_coords(coords, chromosomes, seq_pos_dict):
  """
  Exctract coords for multiple chromosomes stored in a single array into
  a dictionary, keyed by chromosome name. The chromosomes argument is required
  to get the correct array storage order.
  """

  chromo_num_particles = [len(seq_pos_dict[chromo]) for chromo in chromosomes]
  n_seq_pos = sum(chromo_num_particles)
  n_models, n_particles, dims = coords.shape

  if n_seq_pos != n_particles:
    msg = ('Model coordinates must be an array of num models x %d, not %d' %
           (n_seq_pos, n_particles))
    raise(Exception(msg))

  coords_dict = {}

  j = 0
  for i, chromo in enumerate(chromosomes):
    span = chromo_num_particles[i]
    coords_dict[chromo] = coords[:,j:j+span] # all models, slice
    j += span

  return coords_dict


def calc_bins(chromo_limits, particle_size):
  from numpy import arange
  from math import ceil

  bins = {}
  for chr, (start, end) in chromo_limits.items():
    start = (start // particle_size) * particle_size
    end = (end // particle_size + bool(end % particle_size)) * particle_size
    # TODO (kjw53): Original uses ceil, then adds additional particle. Why?
    end += particle_size

    bins[chr] = arange(start, end, particle_size, dtype='int32')
  return bins


def calc_ambiguity_offsets(groups):
  """
  Convert (sorted) ambiguity groups to group-offsets for
  annealing calculations.
  """
  from numpy import arange, zeros

  offsets = arange(len(groups) + 1, dtype='int32')
  group_starts = zeros(len(offsets), dtype='bool')
  group_starts[-1] = group_starts[0] = 1
  group_starts[:-1] |= groups == 0
  group_starts[1:-1] |= groups[1:] != groups[:-1]
  return offsets[group_starts]


def backbone_restraints(seq_pos, particle_size, scale=1.0, lower=0.1, upper=1.1, weight=1.0):
  from numpy import empty, arange, array

  restraints = empty(len(seq_pos) - 1, dtype=Restraint)
  offsets = array([0, 1], dtype='int')
  restraints['indices'] = arange(len(restraints))[:, None] + offsets

  # Normally 1.0 for regular sized particles
  bounds = array([lower, upper], dtype='float') * scale
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


def concatenate_restraints(restraint_dict, pos_dict):
  from itertools import accumulate, chain
  from numpy import concatenate, array
  import operator as op

  chromosome_lengths = map(len, map(pos_dict.__getitem__, sorted(pos_dict)))
  chromosome_offsets = dict(zip(sorted(pos_dict),
                                chain([0], accumulate(chromosome_lengths, op.add))))
  flat_restraints = sorted(flatten_dict(restraint_dict).items())

  particle_indices = concatenate([
    restraints['indices'] + array([[chromosome_offsets[c] for c in chrs]], dtype='int32')
    for chrs, restraints in flat_restraints
  ])
  distances, ambiguity_groups, weights = map(concatenate, (
    [restraints[k] for _, restraints in flat_restraints]
    for k in ['dists', 'ambiguity', 'weight']
  ))

  return particle_indices, distances, weights, ambiguity_groups


def calc_restraints(contact_dict, pos_dict,
                    scale=1.0, lower=0.8, upper=1.2, weight=1.0):
  from numpy import array, searchsorted
  from collections import defaultdict, Counter

  dists = array([[lower, upper]]) * scale

  group_idxs = defaultdict(list)
  # Group indices by ambiguity group
  for (chr_a, chr_b), (pos_a, pos_b, ambig) in flatten_dict(contact_dict).items():
    idxs_a = searchsorted(pos_dict[chr_a], pos_a)
    idxs_b = searchsorted(pos_dict[chr_b], pos_b)
    for ambig, idx_a, idx_b in zip(ambig, idxs_a, idxs_b):
      group_idxs[ambig].append(tuple(sorted([(chr_a, idx_a), (chr_b, idx_b)])))

  # Merge identical ambiguity groups
  group_counts = Counter(map(tuple, map(sorted, group_idxs.values())))

  # Unpack into restraints by chromosome
  restraints = defaultdict(list)
  for ambig, (ends, n) in enumerate(sorted(group_counts.items()), start=1):
    for (chr_a, idx_a), (chr_b, idx_b) in map(tuple, ends):
      restraints[chr_a, chr_b].append(((idx_a, idx_b), dists, ambig, n * weight))

  r = defaultdict(dict)
  for (chr_a, chr_b), rests in restraints.items():
    r[chr_a][chr_b] = array(rests, dtype=Restraint)
  return dict(r)


def anneal_genome(contact_dict, particle_size, prev_seq_pos_dict=None, start_coords=None,
                  contact_dist=(0.8, 1.2), backbone_dist=(0.1, 1.1),
                  temp_range=(5000.0, 10.0), temp_steps=500, dynamics_steps=100, time_step=0.001,
                  random_seed=None, random_radius=10.0, num_models=1):
    """
    Use chromosome contact data to generate distance restraints and then
    apply a simulated annealing protocul to generate/refine coordinates.
    Starting coordinates may be random of from a previous (e.g. lower
    resolution) stage.
    """

    from numpy import (int32, ones, empty, random, concatenate, stack, argsort, geomspace,
                       linspace, arctan, full)
    from math import log, exp, pi
    from functools import partial
    import gc

    bead_size = particle_size ** (1/3)

    if random_seed is not None:
      random.seed(random_seed)
    particle_size = int32(particle_size)
    seq_pos_dict = calc_bins(calc_limits(contact_dict), particle_size)
    chromosomes = sorted(seq_pos_dict)

    # Calculate distance restrains from contact data
    restraint_dict = calc_restraints(contact_dict, seq_pos_dict, scale=bead_size,
                                     lower=contact_dist[0], upper=contact_dist[1])

    # Adjust to keep force/particle approximately constant
    dist = 215.0 * (sum(map(len, seq_pos_dict.values())) /
                    sum(map(lambda v: v['weight'].sum(),
                            flatten_dict(restraint_dict).values())))

    for chr in chromosomes:
      backbone = backbone_restraints(seq_pos_dict[chr], particle_size, bead_size,
                                     backbone_dist[0], backbone_dist[1])
      restraint_dict[chr][chr] = concatenate([backbone, restraint_dict[chr][chr]])

    coords = start_coords or {}
    for chr in chromosomes:
      if chr not in coords:
        coords[chr] = get_random_coords(
          (num_models, len(seq_pos_dict[chr]), 3), random_radius * bead_size
        )
      elif coords[chr].shape[1] != len(seq_pos_dict[chr]):
        coords[chr] = stack([get_interpolated_coords(coords[chr][i], seq_pos_dict[chr],
                                                     prev_seq_pos_dict[chr])
                             for i in range(num_models)])

    # Equal unit masses and radii for all particles
    masses = {chr: ones(len(pos), float) for chr, pos in seq_pos_dict.items()}
    radii = {chr: full(len(pos), bead_size, float)
             for chr, pos in seq_pos_dict.items()}
    repDists = {chr: r * 0.75 for chr, r in radii.items()}

    # Concatenate chromosomal data into a single array of particle restraints
    # for structure calculation.
    restraint_indices, restraint_dists, restraint_weights, ambiguity_groups = (
      concatenate_restraints(restraint_dict, seq_pos_dict)
    )
    coords = concatenate([coords[chr] for chr in chromosomes], axis=1)
    masses = concatenate([masses[chr] for chr in chromosomes])
    radii = concatenate([radii[chr] for chr in chromosomes])
    repDists = concatenate([repDists[chr] for chr in chromosomes])

    restraint_order = argsort(ambiguity_groups)
    restraint_indices = restraint_indices[restraint_order]
    restraint_dists = restraint_dists[restraint_order]
    restraint_weights = restraint_weights[restraint_order]
    ambiguity = calc_ambiguity_offsets(ambiguity_groups[restraint_order])

    # Setup annealig schedule: setup temps and repulsive terms
    temps = geomspace(temp_range[0], temp_range[1], temp_steps, endpoint=False)
    temps *= bead_size ** 2
    repulses = arctan(linspace(0, 1, temp_steps, endpoint=False) * 20 - 10) / pi / arctan(10) + 0.5
    repulses /= bead_size ** 2

    # Update coordinates in the annealing schedule
    time_taken = 0.0

    for model_coords in coords: # For each repeat calculation
      for temp, repulse in zip(temps, repulses):
        gc.collect() # Try to free some memory

        # Update coordinates for this temp
        dt = runDynamics(
          model_coords, masses, radii, repDists,
          restraint_indices, restraint_dists, restraint_weights, ambiguity,
          temp, time_step, dynamics_steps, repulse, dist
        )

        time_taken += dt

      # Center
      model_coords -= model_coords.mean(axis=0)

    # Convert from single coord array to dict keyed by chromosome
    coords_dict = unpack_chromo_coords(coords, chromosomes, seq_pos_dict)

    return coords_dict, seq_pos_dict, restraint_dict


def hierarchical_annealing(contacts, particle_sizes, **kwargs):
    # Initial coords will be random
    start_coords = None

    # Record partile positions from previous stages
    # so that coordinates can be interpolated to higher resolution
    prev_seq_pos = None

    for stage, particle_size in enumerate(particle_sizes):

        print("Running structure caculation stage %d (%d kb)" % (stage+1, (particle_size/1e3)))

        # Can remove large violations (noise contacts inconsistent with structure)
        # once we have a resonable resolution structure
        if stage == 4:
            remove_violated_contacts(contacts, coords_dict, particle_seq_pos,
                                     particle_size, threshold=6.0)
        elif stage == 5:
            remove_violated_contacts(contacts, coords_dict, particle_seq_pos,
                                     particle_size, threshold=5.0)

        coords_dict, particle_seq_pos, restraint_dict = anneal_genome(
          contacts, particle_size, prev_seq_pos, start_coords, **kwargs
        )

        # Next stage based on previous stage's 3D coords
        # and thier respective seq. positions
        start_coords = coords_dict
        prev_seq_pos = particle_seq_pos

    return coords_dict, particle_seq_pos, restraint_dict


def main(args=None):
    from argparse import ArgumentParser
    from sys import argv

    parser = ArgumentParser(description="Calculate a structure from a contact file.")
    parser.add_argument("contacts", type=Path, help="The .ncc file to load contacts from")
    parser.add_argument("output", type=Path, help="The .nuc file to save the structure in")
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

    args = parser.parse_args(argv[1:] if args is None else args)

    contacts = load_ncc_file(str(args.contacts))
    contacts = remove_isolated_contacts(contacts, threshold=args.isolated_threshold)

    coords, seq_pos, restraints = hierarchical_annealing(
        contacts, args.particle_sizes,
        contact_dist=args.contact_dist, backbone_dist=args.backbone_dist,
        # Cautious annealing parameters
        # Don' need to be fixed, but are for simplicity
        temp_range=args.temp_range, temp_steps=args.temp_steps,
        dynamics_steps=args.dyn_steps, time_step=args.time_step,
        # To set up starting coords
        random_seed=args.seed, random_radius=args.random_radius,
        num_models=args.models,
    )

    export_nuc_coords(str(args.output), coords, seq_pos, restraints, vars(args))
    print('Saved structure file to: %s' % str(args.output))

if __name__ == "__main__":
    main()
