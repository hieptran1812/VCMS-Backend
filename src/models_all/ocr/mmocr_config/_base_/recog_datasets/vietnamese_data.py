# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, Syn90k

train_root = '/data/dataset/OCR_IE/wordbase/train'
test_root = '/data/dataset/OCR_IE/wordbase/val'

path1 = f'{train_root}/vn_synthdata'
path2 = f'{train_root}/bill_word'
path3 = f'{train_root}/bkai'
path4 = f'{train_root}/vintext'

test_path1 = f'{test_root}/vn_synthdata'


train1 = dict(
    type='OCRDataset',
    img_prefix=path1,
    ann_file=path1,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(
            type='LineJsonParser',
            keys=['filename', 'text'],
            )),
    pipeline=None,
    test_mode=False)

train2 = dict(
    type='OCRDataset',
    img_prefix=path2,
    ann_file=path2,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(
            type='LineJsonParser',
            keys=['filename', 'text'],
            )),
    pipeline=None,
    test_mode=False)


train3 = dict(
    type='OCRDataset',
    img_prefix=path4,
    ann_file=path4,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(
            type='LineJsonParser',
            keys=['filename', 'text'],
            )),
    pipeline=None,
    test_mode=False)


train4 = dict(
    type='OCRDataset',
    img_prefix=path4,
    ann_file=path4,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(
            type='LineJsonParser',
            keys=['filename', 'text'],
            )),
    pipeline=None,
    test_mode=False)

test1 = dict(
    type='OCRDataset',
    img_prefix=test_path1,
    ann_file=test_path1,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(
            type='LineJsonParser',
            keys=['filename', 'text'],
            )),
    pipeline=None,
    test_mode=False)

train_list = [train1, train2, train3, train4]
test_list = [test1]