from nuc_cython import (runDynamics, getSupportedPairs, calc_restraints, Restraint,
                        concatenate_restraints, getInterpolatedCoords)

def load_ncc_file(file_path):
  """Load chromosome and contact data from NCC format file, as output from NucProcess"""

  from numpy import array

  if file_path.endswith('.gz'):
    import gzip
    file_obj = gzip.open(file_path)

  else:
    file_obj = open(file_path)

  # Observations are treated individually in single-cell Hi-C,
  # i.e. no binning, so num_obs always 1 for each contact
  num_obs = 1

  contact_dict = {}

  for line in file_obj:
    chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, ambig_group, pair_id, swap_pair = line.split()

    if strand_a == '+':
      pos_a = int(f_start_a)
    else:
      pos_a = int(f_end_a)

    if strand_b == '+':
      pos_b = int(f_start_b)
    else:
      pos_b = int(f_end_b)

    if chr_a > chr_b:
      chr_a, chr_b = chr_b, chr_a
      pos_a, pos_b = pos_b, pos_a

    if chr_a not in contact_dict:
      contact_dict[chr_a] = {}

    if chr_b not in contact_dict[chr_a]:
      contact_dict[chr_a][chr_b] = []

    contact_dict[chr_a][chr_b].append((pos_a, pos_b, num_obs, int(ambig_group)))

  file_obj.close()

  for chr_a in contact_dict:
    for chr_b in contact_dict[chr_a]:
      contact_dict[chr_a][chr_b] = array(contact_dict[chr_a][chr_b]).T
  return contact_dict


def calc_limits(contact_dict):
  chromo_limits = {}

  for chr_a in contact_dict:
    for chr_b in contact_dict[chr_a]:
      contacts = contact_dict[chr_a][chr_b]
      if contacts.shape[1] < 1:
        continue

      seq_pos_a = contacts[0]
      seq_pos_b = contacts[1]

      min_a = min(seq_pos_a)
      max_a = max(seq_pos_a)
      min_b = min(seq_pos_b)
      max_b = max(seq_pos_b)

      if chr_a in chromo_limits:
        prev_min, prev_max = chromo_limits[chr_a]
        chromo_limits[chr_a] = [min(prev_min, min_a), max(prev_max, max_a)]
      else:
        chromo_limits[chr_a] = [min_a, max_a]

      if chr_b in chromo_limits:
        prev_min, prev_max = chromo_limits[chr_b]
        chromo_limits[chr_b] = [min(prev_min, min_b), max(prev_max, max_b)]
      else:
        chromo_limits[chr_b] = [min_b, max_b]
  return chromo_limits


def export_pdb_coords(file_path, coords_dict, seq_pos_dict, particle_size, scale=1.0, extended=True):

  from numpy import array, uint32, float32

  alc = ' '
  ins = ' '
  prefix = 'HETATM'
  line_format = '%-80.80s\n'

  if extended:
    pdb_format = '%-6.6s%5.1d %4.4s%s%3.3s %s%4.1d%s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2.2s  %10d\n'
    ter_format = '%-6.6s%5.1d      %s %s%4.1d%s                                                     %10d\n'
  else:
    pdb_format = '%-6.6s%5.1d %4.4s%s%3.3s %s%4.1d%s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2.2s  \n'
    ter_format = '%-6.6s%5.1d      %s %s%4.1d%s                                                     \n'

  file_obj = open(file_path, 'w')
  write = file_obj.write

  chromosomes = list(sorted(seq_pos_dict.keys()))
  num_models = len(coords_dict[chromosomes[0]])
  title = 'NucDynamics genome structure export'

  write(line_format % 'TITLE     %s' % title)
  write(line_format % 'REMARK 210')
  write(line_format % 'REMARK 210 Atom type C is used for all particles')
  write(line_format % 'REMARK 210 Atom number increases every %s bases' % particle_size)
  write(line_format % 'REMARK 210 Residue code indicates chromosome')
  write(line_format % 'REMARK 210 Residue number represents which sequence Mb the atom is in')
  write(line_format % 'REMARK 210 Chain letter is different every chromosome, where Chr1=a, Chr2=b etc.')

  if extended:
    file_obj.write(line_format % 'REMARK 210 Extended PDB format with particle seq. pos. in last column')

  file_obj.write(line_format % 'REMARK 210')

  pos_chromo = {}

  for m in range(num_models):
    line = 'MODEL     %4d' % (m+1)
    write(line_format  % line)

    c = 0
    j = 1
    seqPrev = None
    for chromo in chromosomes:

      if chromo.isdigit():
        idx = int(chromo) - 1
        chain_code = chr(ord('A')+idx)

      elif len(chromo) == 1:
        chain_code = chromo.upper()

      else:
        idx = chromosomes.index(chromo)
        chain_code = chr(ord('a')+idx)

      tlc = chromo
      while len(tlc) < 2:
        tlc = '_' + tlc

      if len(tlc) == 2:
        tlc = 'C' + tlc

      if len(tlc) > 3:
        tlc = tlc[:3]

      chromo_model_coords = coords_dict[chromo][m]

      if not len(chromo_model_coords):
        continue

      pos = seq_pos_dict[chromo]

      for i, seqPos in enumerate(pos):
        c += 1

        seqMb = int(seqPos//1e6) + 1

        if seqMb == seqPrev:
          j += 1
        else:
          j = 1

        el = 'C'
        a = 'C%d' % j

        aName = '%-3s' % a
        x, y, z = chromo_model_coords[i] #XYZ coordinates

        seqPrev = seqMb
        pos_chromo[c] = chromo

        if extended:
          line  = pdb_format % (prefix,c,aName,alc,tlc,chain_code,seqMb,ins,x,y,z,0.0,0.0,el,seqPos)
        else:
          line  = pdb_format % (prefix,c,aName,alc,tlc,chain_code,seqMb,ins,x,y,z,0.0,0.0,el)

        write(line)

    write(line_format  % 'ENDMDL')

  for i in range(c-2):
     if pos_chromo[i+1] == pos_chromo[i+2]:
       line = 'CONECT%5.1d%5.1d' % (i+1, i+2)
       write(line_format  % line)

  write(line_format  % 'END')
  file_obj.close()



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
        coord_data_a = getInterpolatedCoords(coords_a[m], contact_pos_a, particle_seq_pos[chr_a])
        coord_data_b = getInterpolatedCoords(coords_b[m], contact_pos_b, particle_seq_pos[chr_b])

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
  a sphere of given radius
  """

  from numpy import random
  from numpy.linalg import norm

  u = random.uniform(size=shape)
  x = random.normal(size=shape + (3,))
  scaling = (radius * u ** (1/3)) / norm(x, axis=-1)
  return scaling[..., None] * x


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


def backbone_restraints(seq_pos, particle_size, lower=0.1, upper=1.1):
  from numpy import empty, arange, array

  restraints = empty(len(seq_pos) - 1, dtype=Restraint)
  offsets = array([0, 1], dtype='int')
  restraints['indices'] = arange(len(restraints))[:, None] + offsets

  # Normally 1.0 for regular sized particles
  bounds = array([lower, upper], dtype='float')
  restraints['dists'] = ((seq_pos[1:] - seq_pos[:-1]) / particle_size)[:, None] * bounds
  restraints['ambiguity'] = 0 # Use '0' to represent no ambiguity

  return restraints


def anneal_genome(contact_dict, num_models, particle_size,
                  prev_seq_pos_dict=None, start_coords=None,
                  contact_dist=(0.8, 1.2), backbone_dist=(0.1, 1.1),
                  temp_range=(5000.0, 10.0), temp_steps=500,
                  dynamics_steps=100, time_step=0.001,
                  random_seed=None, random_radius=10.0):
    """
    Use chromosome contact data to generate distance restraints and then
    apply a simulated annealing protocul to generate/refine coordinates.
    Starting coordinates may be random of from a previous (e.g. lower
    resolution) stage.
    """

    from numpy import int32, ones, empty, random, concatenate, stack, argsort
    from math import log, exp, atan, pi
    from functools import partial
    import gc

    if random_seed is not None:
      random.seed(random_seed)
    particle_size = int32(particle_size)
    seq_pos_dict = calc_bins(calc_limits(contact_dict), particle_size)
    chromosomes = sorted(seq_pos_dict)

    # Calculate distance restrains from contact data
    restraint_dict = calc_restraints(contact_dict, seq_pos_dict, particle_size, scale=1.0,
                                     lower=contact_dist[0], upper=contact_dist[1])

    for chr in chromosomes:
      backbone = backbone_restraints(seq_pos_dict[chr], particle_size,
                                     backbone_dist[0], backbone_dist[1])
      restraint_dict[chr][chr] = concatenate([backbone, restraint_dict[chr][chr]])

    coords = start_coords or {}
    for chr in chromosomes:
      if chr not in coords:
        coords[chr] = get_random_coords((num_models, len(seq_pos_dict[chr])), random_radius)
      elif coords[chr].shape[1] != len(seq_pos_dict[chr]):
        coords[chr] = stack([getInterpolatedCoords(coords[chr][i],
                                                   seq_pos_dict[chr],
                                                   prev_seq_pos_dict[chr])
                             for i in range(num_models)])

    # Equal unit masses and radii for all particles
    masses = {chr: ones(len(pos), float) for chr, pos in seq_pos_dict.items()}
    radii = {chr: ones(len(pos), float) for chr, pos in seq_pos_dict.items()}

    # Concatenate chromosomal data into a single array of particle restraints
    # for structure calculation.
    restraint_indices, restraint_dists, ambiguity_groups = concatenate_restraints(restraint_dict, seq_pos_dict)
    coords = concatenate([coords[chr] for chr in chromosomes], axis=1)
    masses = concatenate([masses[chr] for chr in chromosomes])
    radii = concatenate([radii[chr] for chr in chromosomes])

    restraint_order = argsort(ambiguity_groups)
    restraint_indices = restraint_indices[restraint_order]
    restraint_dists = restraint_dists[restraint_order]
    ambiguity = calc_ambiguity_offsets(ambiguity_groups[restraint_order])

    # Setup annealig schedule: setup temps and repulsive terms
    adj = 1.0 / atan(10.0)
    decay = log(temp_range[0]/temp_range[1])
    anneal_schedule = []

    for step in range(temp_steps):
      frac = step/float(temp_steps)

      # exponential temp decay
      temp = temp_range[0] * exp(-decay*frac)

      # sigmoidal repusion scheme
      repulse = 0.5 + adj * atan(frac*20.0-10) / pi

      anneal_schedule.append((temp, repulse))

    # Update coordinates in the annealing schedule
    time_taken = 0.0

    for model_coords in coords: # For each repeat calculation
      for temp, repulse in anneal_schedule:
        gc.collect() # Try to free some memory

        # Update coordinates for this temp
        dt = runDynamics(model_coords, masses, radii, restraint_indices, restraint_dists,
                         ambiguity, temp, time_step, dynamics_steps, repulse)

        time_taken += dt

      # Center
      model_coords -= model_coords.mean(axis=0)

    # Convert from single coord array to dict keyed by chromosome
    coords_dict = unpack_chromo_coords(coords, chromosomes, seq_pos_dict)

    return coords_dict, seq_pos_dict


def hierarchical_annealing(contacts, particle_sizes, num_models=1, *args, **kwargs):
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

        coords_dict, particle_seq_pos = anneal_genome(
          contacts, num_models, particle_size, prev_seq_pos, start_coords,
          *args, **kwargs
        )

        # Next stage based on previous stage's 3D coords
        # and thier respective seq. positions
        start_coords = coords_dict
        prev_seq_pos = particle_seq_pos

    return coords_dict, particle_seq_pos


def main(args=None):
    from argparse import ArgumentParser
    from sys import argv
    from pathlib import Path

    parser = ArgumentParser(description="Calculate a structure from a contact file.")
    parser.add_argument("contacts", type=Path, help="The .ncc file to load contacts from")
    parser.add_argument("output", type=Path, help="The .pdb file to save the structure in")
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

    coords, seq_pos = hierarchical_annealing(
        contacts, args.particle_sizes, args.models,
        contact_dist=args.contact_dist, backbone_dist=args.backbone_dist,
        # Cautious annealing parameters
        # Don' need to be fixed, but are for simplicity
        temp_range=args.temp_range, temp_steps=args.temp_steps,
        dynamics_steps=args.dyn_steps, time_step=args.time_step,
        # To set up starting coords
        random_seed=args.seed, random_radius=args.random_radius,
    )

    # Save final coords as PDB format file
    export_pdb_coords(str(args.output), coords, seq_pos, args.particle_sizes[-1])
    print('Saved structure file to: %s' % str(args.output))

if __name__ == "__main__":
    main()
