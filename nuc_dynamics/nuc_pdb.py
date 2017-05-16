#!/usr/bin/env python3
from pathlib import Path
from itertools import chain

def load_nuc_file(file_path, structure="0"):
    import h5py

    with h5py.File(file_path, "r") as f:
        structure = f["structures"][structure]
        coords = {k: v[()] for k, v in structure["coords"].items()}
        seq_pos = {k: v["positions"][()] for k, v in structure["particles"].items()}
        particle_size = structure['calculation'].attrs["particle_sizes"][-1]

    return coords, seq_pos, particle_size

class Column:
    def __init__(self, *args):
        self.text = self.formatter(*args)
        if len(self) > self.length:
            raise ValueError(
                "Formatted text too long ({} chars instead of {})"
                .format(len(self), self.length)
            )

    @property
    def length(self):
        # Columns are easier to read from spec
        start, end = self.columns
        return end - start + 1

    def formatter(self, *args):
        # Auto-generate most common case
        fmt = "{{:{:d}{:s}}}".format(self.length, self.format_code)
        return fmt.format(*args)

    def __str__(self):
        return self.text
    def __len__(self):
        return len(str(self))

class PositiveNumberColumn(Column):
    format_code = "d"
    def __init__(self, num):
        if num <= 0:
            raise ValueError("Number must be > 0, not '{}'".format(num))
        super().__init__(num)

class RecordName(Column):
    columns = (1, 6)
    format_code = "s"
class TitleColumn(Column):
    columns = (11, 80)
    format_code = "s"
class Continuation(PositiveNumberColumn):
    columns = (9, 10)
    def formatter(self, number):
        if number == 1:
            return ' ' * self.length
        else:
            return "{:2d}".format(number)
class Serial(PositiveNumberColumn):
    columns = (7, 11)
class AtomName(Column):
    columns = (13, 16)
    formatter = "{:>2s}{:<2s}".format
    def __init__(self, elem, suffix):
        if len(elem) > 2 or len(suffix) > 2:
            raise ValueError("Element name or suffix too long (> 2 chars)")
        super().__init__(elem, suffix)
class AlternateLocation(Column):
    columns = (17, 17)
    format_code = "s"
class ResidueName(Column):
    columns = (18, 20)
    format_code = "s"
class ChainID(Column):
    columns = (22, 22)
    format_code = "s"
    def __init__(self, chain_id):
        if chain_id == ' ':
            raise ValueError("Chain identifier may not be blank")
        super().__init__(chain_id)
class ResidueSequence(PositiveNumberColumn):
    columns=(23, 26)
class InsertionCode(Column):
    columns = (27, 27)
    format_code = "s"
class X(Column):
    columns = (31, 38)
    formatter = "{:8.3f}".format
class Y(Column):
    columns = (39, 46)
    formatter = "{:8.3f}".format
class Z(Column):
    columns = (47, 54)
    formatter = "{:8.3f}".format
class Occupancy(Column):
    columns = (55, 60)
    formatter = "{:6.2f}".format
class Temp(Column):
    columns = (61, 66)
    formatter = "{:6.2f}".format
class Element(Column):
    columns = (77, 78)
    formatter = "{:>2s}".format
class Charge(Column):
    columns = (79, 80)
    def formatter(self, charge):
        if charge == 0:
            return ' ' * self.length
        else:
            return "{:2d}".format(charge)
class RemarkNumber(PositiveNumberColumn):
    columns = (8, 10)
class RemarkColumn(Column):
    columns = (12, 79)
    format_code = "s"
class ModelNumber(PositiveNumberColumn):
    columns = (11, 14)
class ConnectStart(PositiveNumberColumn):
    columns = (7, 11)
class ConnectEnd(PositiveNumberColumn):
    columns = (12, 16)
class SeqPos(Column):
    columns = (81, 90)
    format_code = "d"
    def __init__(self, num):
        if num < 0:
            raise ValueError("Sequenc position must be >= 0, not '{}'".format(num))
        super().__init__(num)

class Record:
    line_length = 80

    def __init__(self, *args):
        columns = tuple(c(*a) if isinstance(a, tuple) else c(a)
                        for c, a in zip(self.column_types, args))
        self.columns = (RecordName(self.record_name),) + columns

    @property
    def padding(self):
        ends = (c.columns[1] for c in chain((RecordName,), self.column_types))
        starts = chain((c.columns[0] for c in self.column_types), (81,))
        return [' ' * (e - s - 1) for e, s in zip(starts, ends)]

    def __str__(self):
        text = ''.join(chain.from_iterable(zip(map(str, self.columns), self.padding)))
        if len(text) != self.line_length:
            raise ValueError("Formatted line has incorrect length ({} chars)"
                             .format(len(text)))
        return text

class Title(Record):
    column_types = (Continuation, TitleColumn)
    record_name = "TITLE"
    def __init__(self, text, num=1):
        super().__init__(num, text)

class HetAtom(Record):
    column_types = (
        Serial, AtomName, AlternateLocation, ResidueName, ChainID, ResidueSequence,
        InsertionCode, X, Y, Z, Occupancy, Temp, Element, Charge
    )
    record_name = "HETATM"
    def __init__(self, serial, atom_name='', altloc='', residue='', chain='', sequence=1,
                 insertion_code='', coords=(0, 0, 0), occupancy=1.0, temp=0.0, charge=0):
        x, y, z = coords
        element = atom_name[0]
        super().__init__(
            serial, atom_name, altloc, residue, chain, sequence, insertion_code,
            x, y, z, occupancy, temp, element, charge
        )

class Remark(Record):
    column_types = (RemarkNumber, RemarkColumn)
    record_name = "REMARK"

class Model(Record):
    column_types = (ModelNumber,)
    record_name = "MODEL"

class EndModel(Record):
    column_types = ()
    record_name = "ENDMDL"

class Connect(Record):
    # Not strictly correct, can have up to 4 ends
    column_types = (ConnectStart, ConnectEnd)
    record_name = "CONECT"

class End(Record):
    column_types = ()
    record_name = "END"

class ExtendedHetAtom(HetAtom):
    column_types = HetAtom.column_types + (SeqPos,)
    line_length = 90
    def __init__(self, serial, atom_name='', altloc='', residue='', chain='', sequence=1,
                 insertion_code='', coords=(0, 0, 0), occupancy=1.0, temp=0.0, charge=0,
                 seq_pos=0):
        x, y, z = coords
        element = atom_name[0]
        Record.__init__(
            self, serial, atom_name, altloc, residue, chain, sequence, insertion_code,
            x, y, z, occupancy, temp, element, charge, seq_pos
        )

def export_pdb_coords(file_path, coords_dict, seq_pos_dict, particle_size,
                      scale=1.0, extended=True):

    from numpy import array, uint32, float32
    from string import ascii_uppercase, ascii_lowercase
    from collections import OrderedDict
    from itertools import accumulate, chain

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
