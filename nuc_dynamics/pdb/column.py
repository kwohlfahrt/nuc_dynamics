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
