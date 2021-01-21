import pathlib
from mrbfile import MRBFile
from taggedarrays import SegmentInfo, SegmentationData


jh_manseg_fpath = 'G:/Cochlea/Data_TPL_Library/johannes_manual_segmentation/44.mrb'
local_tst_fpath = 'C:/Users/Jannik/Desktop/mrbreader/tests/assets/testmrb_multiseg.mrb'

fpath = pathlib.Path(jh_manseg_fpath)

mrb = MRBFile(fpath)

print('Getting memeber infos;  ')
data_members = mrb.get_member_info()
print(mrb.raw_members)


print('Reading segmentations')
segdata = mrb.read_segmentations()[0]

rawdata = mrb.read_raws()[0]

print(rawdata.metadata)

print(segdata.metadata)

raise Exception


print('SegmentationData seginfos dict:')
for k, v in segdata.seginfos.items():
    print(f'Key: {k}  -  value {v}')

seginfos = SegmentInfo.from_header(segheader)

for seginfo in seginfos:
    print(f'Segment {seginfo.name} with color {seginfo.color}')