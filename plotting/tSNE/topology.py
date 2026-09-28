"""Decode the exact decimal topology format without float32 precision loss.

Truth labels are metadata only: never pass these codes or category IDs to PCA,
t-SNE, a scaler, or the network's input tensors.
"""
from decimal import Decimal, InvalidOperation
from functools import lru_cache

CATEGORIES = ('0p1pi', '1p0pi', '2p0pi', '1p1pi', '2p1pi',
              '1p2pi', '2p2pi', 'NpNpi', 'Other')


@lru_cache(maxsize=8192)
def decode(value):
    """Return (canonical_code, proton_count, total_pion_count, category).

    C++ layout: flavor + PP + 00 + pi+pi+ + 00 + pi-pi- + 00 + pi0pi0.
    Decimal accepts exact integers written with '.0' or scientific notation.
    Reject malformed separators rather than silently interpreting rounded labels.
    """
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError(f'Nonintegral topology: {value}')
        code = str(int(number))
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid topology: {value}') from exc
    if (len(code) != 15 or code[0] not in '123456'
            or any(code[i:i+2] != '00' for i in (3, 7, 11))):
        raise ValueError(f'Malformed or precision-damaged topology: {value}')
    protons = int(code[1:3])
    pions = sum(int(code[i:i+2]) for i in (5, 9, 13))
    # User's revised definition: ANY multiplicity >= 3 goes into this one bin.
    if protons >= 3 or pions >= 3:
        category = 'NpNpi'
    else:
        candidate = f'{protons}p{pions}pi'
        category = candidate if candidate in CATEGORIES else 'Other'
    return code, protons, pions, category
