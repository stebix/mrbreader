import pathlib
from reader import MRBFile



fpath = pathlib.Path('C:/Users/Jannik/Desktop/mrbreader/tests/assets/testmrb_multiseg.mrb')

mrb = MRBFile(fpath)

data_members = mrb.get_member_info()

print(mrb.raw_members)