import pytest

def test_entry_padding():
    from nuc_dynamics.pdb.record import Title
    assert Title("Foo").padding == ['  ', '', '']


def test_title():
    from nuc_dynamics.pdb.record import Title
    assert str(Title("Foo bar baz")) == "{:<80s}".format("TITLE     Foo bar baz")
    assert str(Title("Foo bar baz", 2)) == "{:<80s}".format("TITLE    2Foo bar baz")
    with pytest.raises(ValueError):
        Title("Foo bar baz", 0)


def test_hetatom():
    from nuc_dynamics.pdb.record import HetAtom
    het = HetAtom(
        8238, ("S", ""), residue='SO4', chain='A', sequence=2001,
        coords=[10.8849,-15.746,-14.404444], temp=47.84
    )
    expected = "{:<80s}".format(
        "HETATM 8238  S   SO4 A2001      10.885 -15.746 -14.404  1.00 47.84           S"
    )
    assert str(het) == expected


def test_hetatm_overflow():
    from nuc_dynamics.pdb.record import HetAtom
    HetAtom(8238, ("S", ""), coords=[10.8849,-15.746,1e3],)
    with pytest.raises(ValueError):
        HetAtom(8238, ("S", ""), coords=[10.8849,-15.746,1e4],)
