"""
Homogenization template for 3 class semantic segmentation of human cochlea CT scans.
"""


cochlea_equivalents = frozenset(
    ('cochlea', 'Cochlea', 'chl', 'Schnecke', 'schnecke')
)
vestibulum_equivalents = frozenset(
    ('vestibulum', 'Vestibulum', 'vest')
)
canals_equivalents = frozenset(
    ('Bogengänge', 'bogengänge', 'bogengaenge', 'Bogengaenge',
     'canals', 'Canals', 'semicircular canals', 'Bogen', 'bogen')
)

template = {
    cochlea_equivalents : {
        'name' : 'cochlea',
        'color' : (1, 0, 0),
        'ID' : 'Segment_1',
        'label_value' : 1
    },
    vestibulum_equivalents : {
        'name' : 'vestibulum',
        'color' : (0, 1, 0),
        'ID' : 'Segment_2',
        'label_value' : 2
    },
    canals_equivalents : {
        'name' : 'canals',
        'color' : (0, 0, 1),
        'ID' : 'Segment_3',
        'label_value' : 3
    }
}

