import pytest
import numpy as np

header_template = [
    "TITLE     NucDynamics genome structure export",
    "REMARK 210",
    "REMARK 210 Atom type C is used for all particles",
    "REMARK 210 Atom number increases every {particle_size} bases",
    "REMARK 210 Residue code indicates chromosome",
    "REMARK 210 Residue number represents which sequence Mb the atom is in",
    "REMARK 210 Chain letter is different every chromosome, where Chr1=A, Chr2=B...",
    "REMARK 210 Extended PDB format with particle seq. pos. in last column",
    "REMARK 210",
]

pad = "{:80s}\n".format

def test_export_pdb_coords(tmpdir):
    from nuc_dynamics.nuc_pdb import export_pdb_coords

    p = tmpdir.join('test.pdb')
    coords = {'foo': np.array([[[  0.0, -1.0,  4.0],
                                [ 10.0,  5.0,  8.0],
                                [ 10.0,  5.0,  8.0],],
                               [[  0.5, -1.5,  4.5],
                                [ 10.5,  5.5,  8.5],
                                [ 10.5,  5.5,  8.5],],]),
              '1': np.array([[[  2.0,  3.0, -5.0],
                              [-10.0,  5.0,  0.1],],
                             [[  2.5,  3.5, -5.5],
                              [-10.5,  5.5,  0.6],],])}
    seq_pos = {'foo': np.array([10, 15, 20]), '1': [0, 10]}
    particle_size = 10

    header = [l.format(particle_size=particle_size) for l in header_template]
    header = list(map(pad, header))
    del header[-2]

    expected = [
        "MODEL        1",
        "HETATM    1  C1  C_1 A   1       2.000   3.000  -5.000  0.00  0.00           C",
        "HETATM    2  C2  C_1 A   1     -10.000   5.000   0.100  0.00  0.00           C",
        "HETATM    3  C3  foo b   1       0.000  -1.000   4.000  0.00  0.00           C",
        "HETATM    4  C4  foo b   1      10.000   5.000   8.000  0.00  0.00           C",
        "HETATM    5  C5  foo b   1      10.000   5.000   8.000  0.00  0.00           C",
        "ENDMDL",
        "MODEL        2",
        "HETATM    1  C1  C_1 A   1       2.500   3.500  -5.500  0.00  0.00           C",
        "HETATM    2  C2  C_1 A   1     -10.500   5.500   0.600  0.00  0.00           C",
        "HETATM    3  C3  foo b   1       0.500  -1.500   4.500  0.00  0.00           C",
        "HETATM    4  C4  foo b   1      10.500   5.500   8.500  0.00  0.00           C",
        "HETATM    5  C5  foo b   1      10.500   5.500   8.500  0.00  0.00           C",
        "ENDMDL",
        "CONECT    1    2",
        "CONECT    3    4",
        "CONECT    4    5",
        "END",
    ]
    expected = list(map(pad, expected))

    export_pdb_coords(p, coords, seq_pos, particle_size, extended=False)
    with p.open('r') as f:
        for line in f:
            assert len(line) == 81
            assert line[-1] == '\n'

        f.seek(0)
        for ref, line in zip(header, f):
            assert ref == line

        for ref, line in zip(expected, f):
            assert ref == line

        with pytest.raises(StopIteration):
            next(f)

def test_export_pdb_coords_extended(tmpdir):
    from nuc_dynamics.nuc_pdb import export_pdb_coords

    p = tmpdir.join('test.pdb')
    coords = {'foo': np.array([[[  0.0, -1.0,  4.0],
                                [ 10.0,  5.0,  8.0],
                                [ 10.0,  5.0,  8.0],],
                               [[  0.5, -1.5,  4.5],
                                [ 10.5,  5.5,  8.5],
                                [ 10.5,  5.5,  8.5],],]),
              '1': np.array([[[  2.0,  3.0, -5.0],
                              [-10.0,  5.0,  0.1],],
                             [[  2.5,  3.5, -5.5],
                              [-10.5,  5.5,  0.6],],])}
    seq_pos = {'foo': np.array([10, 15, 20]), '1': [0, 10]}
    particle_size = 10

    header = [l.format(particle_size=particle_size) for l in header_template]
    header = list(map(pad, header))

    expected = [
        "MODEL        1",
        "HETATM    1  C1  C_1 A   1       2.000   3.000  -5.000  0.00  0.00           C           0",
        "HETATM    2  C2  C_1 A   1     -10.000   5.000   0.100  0.00  0.00           C          10",
        "HETATM    3  C3  foo b   1       0.000  -1.000   4.000  0.00  0.00           C          10",
        "HETATM    4  C4  foo b   1      10.000   5.000   8.000  0.00  0.00           C          15",
        "HETATM    5  C5  foo b   1      10.000   5.000   8.000  0.00  0.00           C          20",
        "ENDMDL",
        "MODEL        2",
        "HETATM    1  C1  C_1 A   1       2.500   3.500  -5.500  0.00  0.00           C           0",
        "HETATM    2  C2  C_1 A   1     -10.500   5.500   0.600  0.00  0.00           C          10",
        "HETATM    3  C3  foo b   1       0.500  -1.500   4.500  0.00  0.00           C          10",
        "HETATM    4  C4  foo b   1      10.500   5.500   8.500  0.00  0.00           C          15",
        "HETATM    5  C5  foo b   1      10.500   5.500   8.500  0.00  0.00           C          20",
        "ENDMDL",
        "CONECT    1    2",
        "CONECT    3    4",
        "CONECT    4    5",
        "END",
    ]
    expected = list(map(pad, expected))

    export_pdb_coords(p, coords, seq_pos, particle_size)
    with p.open('r') as f:
        for line in f:
            assert len(line) in (81, 91)
            assert line[-1] == '\n'

        f.seek(0)
        for ref, line in zip(header, f):
            assert ref == line

        for ref, line in zip(expected, f):
            assert ref == line

        with pytest.raises(StopIteration):
            next(f)


def test_entry_padding():
    from nuc_dynamics.nuc_pdb import Title
    assert Title("Foo").padding == ['  ', '', '']


def test_title():
    from nuc_dynamics.nuc_pdb import Title
    assert str(Title("Foo bar baz")) == pad("TITLE     Foo bar baz")[:-1]
    assert str(Title("Foo bar baz", 2)) == pad("TITLE    2Foo bar baz")[:-1]
    with pytest.raises(ValueError):
        Title("Foo bar baz", 0)


def test_hetatom():
    from nuc_dynamics.nuc_pdb import HetAtom
    het = HetAtom(
        8238, ("S", ""), residue='SO4', chain='A', sequence=2001,
        coords=[10.8849,-15.746,-14.404444], temp=47.84
    )
    expected = pad(
        "HETATM 8238  S   SO4 A2001      10.885 -15.746 -14.404  1.00 47.84           S"
    )[:-1]
    assert str(het) == expected


def test_hetatm_overflow():
    from nuc_dynamics.nuc_pdb import HetAtom
    HetAtom(8238, ("S", ""), coords=[10.8849,-15.746,1e3],)
    with pytest.raises(ValueError):
        HetAtom(8238, ("S", ""), coords=[10.8849,-15.746,1e4],)
