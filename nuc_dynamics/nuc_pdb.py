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


def main(args=None):
    from argparse import ArgumentParser
    from sys import argv
    from numpy import concatenate, array

    parser = ArgumentParser(description="Convert a .nuc structure to .pdb")
    parser.add_argument("input", type=Path, help="The .nuc file to load")
    parser.add_argument("output", type=Path, help="The .pdb file to write")
    parser.add_argument("--structure", type=str, default="0",
                        help="The structure to save.")

    args = parser.parse_args(argv[1:] if args is None else args)
    export_pdb_coords(str(args.output), *load_nuc_file(args.input, args.structure))

if __name__ == "__main__":
    main()
