from nuc_cython import (runDynamics, getSupportedPairs, calc_restraints,
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

      seq_pos_a = contacts[1]
      seq_pos_b = contacts[2]

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

  for chromoA in contact_dict:
    for chromoB in contact_dict[chromoA]:
      contacts = contact_dict[chromoA][chromoB]
      positions = array(contacts[:2], int32).T

      if len(positions): # Sometimes empty e.g. for MT, Y chromos
        active_idx = getSupportedPairs(positions, int32(threshold))
        contact_dict[chromoA][chromoB] = contacts[:,active_idx]

  return contact_dict


def remove_violated_contacts(contact_dict, coords_dict, particle_seq_pos, particle_size, threshold=5.0):
  """
  Remove contacts whith structure distances that exceed a given threshold
  """
  from numpy import int32, sqrt

  for chr_a in contact_dict:
    for chr_b in contact_dict[chr_a]:
      contacts = contact_dict[chr_a][chr_b]

      contact_pos_a = contacts[0].astype(int32)
      contact_pos_b = contacts[1].astype(int32)

      coords_a = coords_dict[chr_a]
      coords_b = coords_dict[chr_b]

      struc_dists = []

      for m in range(len(coords_a)):
        coord_data_a = getInterpolatedCoords([chr_a], {chr_a:contact_pos_a}, particle_seq_pos, coords_a[m])
        coord_data_b = getInterpolatedCoords([chr_b], {chr_b:contact_pos_b}, particle_seq_pos, coords_b[m])

        deltas = coord_data_a - coord_data_b
        dists = sqrt((deltas*deltas).sum(axis=1))
        struc_dists.append(dists)

      # Average over all conformational models
      struc_dists = array(struc_dists).T.mean(axis=1)

      # Select contacts with distances below distance threshold
      indices = (struc_dists < threshold).nonzero()[0]
      contact_dict[chr_a][chr_b] = contacts[:,indices]

  return contact_dict


def get_random_coords(pos_dict, chromosomes, num_models, radius=10.0):
  """
  Get random, uniformly sampled coorinate positions, restricted to
  a sphere of given radius
  """

  from numpy import empty, random

  num_particles = sum([len(pos_dict[chromo]) for chromo in chromosomes])
  coords = empty((num_models, num_particles, 3))
  r2 = radius*radius

  for m in range(num_models):

    for i in range(num_particles):
      x = y = z = radius

      while x*x + y*y + z*z >= r2:
        x = random.uniform(-radius, radius)
        y = random.uniform(-radius, radius)
        z = random.uniform(-radius, radius)

      coords[m,i] = [x,y,z]

  return coords


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
    msg = 'Model coordinates must be an array of num models x %d' % (n_seq_pos,)
    raise(Exception(msg))

  coords_dict = {}

  j = 0
  for i, chromo in enumerate(chromosomes):
    span = chromo_num_particles[i]
    coords_dict[chromo] = coords[:,j:j+span] # all models, slice
    j += span

  return coords_dict


def anneal_genome(contact_dict, chromo_limits, num_models, particle_size,
                  general_calc_params, anneal_params,
                  prev_seq_pos_dict=None, start_coords=None):
    """
    Use chromosome contact data to generate distance restraints and then
    apply a simulated annealing protocul to generate/refine coordinates.
    Starting coordinates may be random of from a previous (e.g. lower
    resolution) stage.
    """

    from numpy import int32, ones, empty, random, concatenate
    from math import log, exp, atan, pi
    import gc

    random.seed(general_calc_params['random_seed'])
    particle_size = int32(particle_size)
    chromosomes = sorted(chromo_limits)

    # Calculate distance restrains from contact data
    restraint_dict, seq_pos_dict = calc_restraints(chromosomes, contact_dict, particle_size,
                                                   scale=1.0, exponent=general_calc_params['dist_power_law'],
                                                   lower=general_calc_params['contact_dist_lower'],
                                                   upper=general_calc_params['contact_dist_upper'])

    # Concatenate chromosomal data into a single array of particle restraints
    # for structure calculation. Add backbone restraints between seq. adjasent particles.
    restraint_indices, restraint_dists = concatenate_restraints(restraint_dict, seq_pos_dict, particle_size,
                                                                general_calc_params['backbone_dist_lower'],
                                                                general_calc_params['backbone_dist_upper'])

    # Setup starting structure
    if (start_coords is None) or (prev_seq_pos_dict is None):
      coords = get_random_coords(seq_pos_dict, chromosomes, num_models,
                                 general_calc_params['random_radius'])

      num_coords = coords.shape[1]

    else:
      # Convert starting coord dict into single array
      coords = concatenate([start_coords[chr] for chr in chromosomes], axis=1)
      num_coords = sum([len(seq_pos_dict[c]) for c in chromosomes])

      if coords.shape[1] != num_coords: # Change of particle_size
        interp_coords = empty((num_models, num_coords, 3))

        for m in range(num_models): # Starting coords interpolated from previous particle positions
          interp_coords[m] = getInterpolatedCoords(chromosomes, seq_pos_dict, prev_seq_pos_dict, coords[m])

        coords = interp_coords

    # Equal unit masses and radii for all particles
    masses = ones(num_coords,  float)
    radii = ones(num_coords,  float)

    # Ambiguiity strides not used here, so set to 1
    num_restraints = len(restraint_indices)
    ambiguity = ones(num_restraints,  int32)

    # Below will be set to restrict memory allocation in the repusion list
    # (otherwise all vs all can be huge)
    n_rep_max = int32(0)

    # Annealing parameters
    temp_start = anneal_params['temp_start']
    temp_end = anneal_params['temp_end']
    temp_steps = anneal_params['temp_steps']

    # Setup annealig schedule: setup temps and repulsive terms
    adj = 1.0 / atan(10.0)
    decay = log(temp_start/temp_end)
    anneal_schedule = []

    for step in range(temp_steps):
      frac = step/float(temp_steps)

      # exponential temp decay
      temp = temp_start * exp(-decay*frac)

      # sigmoidal repusion scheme
      repulse = 0.5 + adj * atan(frac*20.0-10) / pi

      anneal_schedule.append((temp, repulse))

    # Paricle dynamics parameters
    # (these need not be fixed for all stages, but are for simplicity)
    dyn_steps = anneal_params['dynamics_steps']
    time_step = anneal_params['time_step']

    # Update coordinates in the annealing schedule
    time_taken = 0.0

    for m in range(num_models): # For each repeat calculation
      model_coords = coords[m]

      for temp, repulse in anneal_schedule:
        gc.collect() # Try to free some memory

        # Update coordinates for this temp
        dt, n_rep_max = runDynamics(model_coords, masses, radii, restraint_indices, restraint_dists,
                                    ambiguity, temp, time_step, dyn_steps, repulse, nRepMax=n_rep_max)

        n_rep_max = int32(1.05 * n_rep_max) # Base on num in prev cycle, plus an overhead
        time_taken += dt

      # Center
      model_coords -= model_coords.mean(axis=0)

      # Update
      coords[m] = model_coords

    # Convert from single coord array to dict keyed by chromosome
    coords_dict = unpack_chromo_coords(coords, chromosomes, seq_pos_dict)

    return coords_dict, seq_pos_dict



from time import time

# Number of alternative conformations to generate from repeat calculations
# with different random starting coordinates
num_models = 2

# Parameters to setup restraints and starting coords
general_calc_params = {'dist_power_law':-0.33,
                       'contact_dist_lower':0.8, 'contact_dist_upper':1.2,
                       'backbone_dist_lower':0.1, 'backbone_dist_upper':1.1,
                       'random_seed':int(time()), 'random_radius':10.0}

# Annealing & dyamics parameters: the same for all stages
# (this is cautious, but not an absolute requirement)
anneal_params = {'temp_start':5000.0, 'temp_end':10.0, 'temp_steps':500,
                 'dynamics_steps':100, 'time_step':0.001}

# Hierarchical scale protocol
particle_sizes = [8e6, 4e6, 2e6, 4e5, 2e5, 1e5]

# Load single-cell Hi-C data from NCC contact file, as output from NucProcess
contact_dict = load_ncc_file('example_chromo_data/P36D6.ncc')
chromo_limits = calc_limits(contact_dict)

# Only use contacts which are supported by others nearby in sequence, in the initial instance
remove_isolated_contacts(contact_dict, threshold=2e6)

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
        remove_violated_contacts(contact_dict, coords_dict, particle_seq_pos,
                                 particle_size, threshold=6.0)
    elif stage == 5:
        remove_violated_contacts(contact_dict, coords_dict, particle_seq_pos,
                                 particle_size, threshold=5.0)

    coords_dict, particle_seq_pos = anneal_genome(contact_dict, chromo_limits, num_models,
                                                  particle_size, general_calc_params, anneal_params,
                                                  prev_seq_pos, start_coords)

    # Next stage based on previous stage's 3D coords
    # and thier respective seq. positions
    start_coords = coords_dict
    prev_seq_pos = particle_seq_pos

# Save final coords as PDB format file
save_path = 'example_chromo_data/P36D6.pdb'
export_pdb_coords(save_path, coords_dict, particle_seq_pos, particle_size)
print('Saved structure file to: %s' % save_path)
