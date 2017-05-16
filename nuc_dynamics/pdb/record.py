from itertools import chain
from .column import *

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
