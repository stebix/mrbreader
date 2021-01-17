import pathlib
from reader import MRBFile
from taggedarrays import SegmentInfo, SegmentationData


fpath = pathlib.Path('C:/Users/Jannik/Desktop/mrbreader/tests/assets/testmrb_multiseg.mrb')

mrb = MRBFile(fpath)

print('Getting memeber infos;  ')
data_members = mrb.get_member_info()
print(mrb.raw_members)


print('Reading segmentations')
segdata_arr, segheader = mrb.read_segmentations()[0]

segdata = SegmentationData(segdata_arr, segheader)
print('SegmentationData seginfos dict:')
for k, v in segdata.seginfos.items():
    print(f'Key: {k}  -  value {v}')

seginfos = SegmentInfo.from_header(segheader)

for seginfo in seginfos:
    print(f'Segment {seginfo.name} with color {seginfo.color}')