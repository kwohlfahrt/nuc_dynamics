#!/usr/bin/env python3
from pathlib import Path

def load_nuc_file(file_path, structure="0"):
    import h5py

    with h5py.File(file_path, "r") as f:
        structure = f["structures"][structure]
        coords = {k: v[()] for k, v in structure["coords"].items()}
        seq_pos = {k: v["positions"][()] for k, v in structure["particles"].items()}
        particle_size = structure['calculation'].attrs["particle_sizes"][-1]

    return coords, seq_pos, particle_size

def export_pdb_coords(file_path, coords_dict, seq_pos_dict, particle_size,
                      scale=1.0, extended=True):

    from numpy import array, uint32, float32
    from string import ascii_uppercase, ascii_lowercase
    from collections import OrderedDict
    from itertools import accumulate, chain
    from .pdb.record import (Title, Remark, Model, EndModel, End, HetAtom, ExtendedHetAtom,
                             Connect)

    alc = ' '
    ins = ' '

    hetatm = ExtendedHetAtom if extended else HetAtom

    with file_path.open('w') as f:
        write = lambda c: f.write(str(c) + '\n')
        coords_dict = OrderedDict(sorted(coords_dict.items()))
        seq_pos_dict = OrderedDict(sorted(seq_pos_dict.items()))
        chromosomes = list(seq_pos_dict.keys())
        num_models = len(next(iter(coords_dict.values())))

        write(Title('NucDynamics genome structure export'))
        write(Remark(210, ""))
        write(Remark(210, "Atom type C is used for all particles"))
        write(Remark(210, "Atom number increases every {:d} bases".format(particle_size)))
        write(Remark(210, "Residue code indicates chromosome"))
        write(Remark(210, "Residue number represents which sequence Mb the atom is in"))
        write(Remark(210, "Chain letter is different every chromosome, where Chr1=A, Chr2=B..."))
        if extended:
            write(Remark(210, "Extended PDB format with particle seq. pos. in last column"))
        write(Remark(210, ""))

        for m in range(num_models):
            write(Model(m+1))

            c = 0
            seqPrev = None
            for chromo in chromosomes:
                if chromo.isdigit():
                    chain_code = ascii_uppercase[int(chromo) - 1]
                elif len(chromo) == 1:
                    chain_code = chromo.upper()
                else:
                    chain_code = ascii_lowercase[chromosomes.index(chromo)]

                resName = chromo[:3]
                if len(chromo) < 2:
                    resName = resName.rjust(2, '_')
                if len(chromo) < 3:
                    resName = 'C' + resName

                pos = seq_pos_dict[chromo]
                coords = coords_dict[chromo][m]
                if len(pos) != len(coords):
                    raise ValueError("Sequence position and coordinates have different length.")

                for i, (coord, seqPos) in enumerate(zip(coords, pos)):
                    c += 1

                    seqMb = int(seqPos//1e6) + 1
                    j = j+1 if seqMb == seqPrev else 1
                    seqPrev = seqMb

                    aName = ('C', str(j))

                    data = [c,aName,alc,resName,chain_code,seqMb,ins,coord,0.0,0.0,0]
                    if extended:
                        data.append(seqPos)
                    write(hetatm(*data))

            write(EndModel())

        chromo_offsets = chain([0], accumulate(map(len, seq_pos_dict.values())))
        chromo_lengths = map(len, seq_pos_dict.values())
        for offset, chromo_len in zip(chromo_offsets, chromo_lengths):
            for i in range(1, chromo_len):
                write(Connect(offset+i, offset+i+1))

        write(End())


def main(args=None):
    from argparse import ArgumentParser
    from sys import argv
    from numpy import concatenate, array

    parser = ArgumentParser(description="Convert a .nuc structure to .pdb")
    parser.add_argument("input", type=Path, help="The .nuc file to load")
    parser.add_argument("output", type=Path, help="The .pdb file to write")
    parser.add_argument("--structure", type=str, default="0",
                        help="The structure to save.")
    parser.add_argument("--scale", type=float, default=0.1,
                        help="The coordinate scaling (to avoid overflow of fixed columns)")

    args = parser.parse_args(argv[1:] if args is None else args)
    coords, seq_pos, particle_size = load_nuc_file(args.input, args.structure)
    coords = {k: v * args.scale for k, v in coords.items()}
    export_pdb_coords(args.output, coords, seq_pos, particle_size)

if __name__ == "__main__":
    main()
