from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # 추가
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate
from utils import dist_util  # 추가
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate

def get_dataset_class(name, load_mode):
    if name == "babel":
        if 'gt' in load_mode or load_mode == 'evaluator_train': # reference motion for evaluation
            from data_loaders.humanml.data.dataset import BABEL_eval
            return BABEL_eval
        elif load_mode == 'gen':
            from data_loaders.amass.babel import BABEL
            return BABEL
        elif load_mode == 'train':
            from data_loaders.amass.babel_flowmdm import BABEL
            return BABEL
        else:
            raise ValueError(f'Unsupported load_mode name [{load_mode}]')
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, load_mode='train'):
    print(name, load_mode)
    if "gt" in load_mode:
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml"]:
        return t2m_collate
    elif name == 'babel' and load_mode != "evaluator_train":
        from data_loaders.tensors import babel_collate
        return babel_collate
    elif name == 'babel':
        from data_loaders.humanml.data.dataset import collate_fn as sorted_collate
        return sorted_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', load_mode='train', **kwargs):
    
    # 1. 데이터셋 클래스 가져오기
    DATA = get_dataset_class(name, load_mode) # HumanML3D 클래스
    pose_rep = kwargs.get('pose_rep', 'rot6d')
    
    # 2.1. humanml인 경우
    if name in ["humanml"]:
        load_mode = "gt" if "gt" in load_mode else load_mode
        dataset = DATA(load_mode, split=split, pose_rep=pose_rep, num_frames=num_frames) # 실제로는: HumanML3D(load_mode='train', split='train', num_frames=False)
    
    # 2.2. babel인 경우
    elif name == "babel":
        cropping_sampler = kwargs.get('cropping_sampler', False)
        opt = kwargs.get('opt', None)
        batch_size = kwargs.get('batch_size', None)
        from data_loaders.amass.transforms import SlimSMPLTransform
        from data_loaders.amass.sampling import FrameSampler
        if ((split=='val') and (cropping_sampler==True)):
            transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True, canonicalize=False)
        else:
            transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True)
        sampler = FrameSampler(min_len=num_frames[0], max_len=num_frames[1])
        dataset = DATA(split=split,
                       datapath='./dataset/babel/babel-smplh-30fps-male',
                       transforms=transform, load_mode=load_mode, opt=opt, sampler=sampler,
                       cropping_sampler=cropping_sampler)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, num_workers=8, split='train', load_mode='train', shuffle=True, drop_last=True, **kwargs):
    
    # 1단계: dataset 객체 생성
    dataset = get_dataset(name, num_frames, split, load_mode, batch_size=batch_size, **kwargs)
    
    # 2단계: collate 함수 선택
    collate = get_collate_fn(name, load_mode)

    # 3단계: DataLoader 생성
    # loader = DataLoader(
    #     dataset, batch_size=batch_size, shuffle=shuffle,
    #     num_workers=num_workers, drop_last=drop_last, collate_fn=collate
    # )
    
    if dist_util.get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # shuffle=True 대신 sampler 사용
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate  # 이 줄 추가!
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate  # 이 줄 추가!
        )

    return loader