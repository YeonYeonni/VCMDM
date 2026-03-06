from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
import pickle

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
import torch

from data_loaders.amass.babel_flowmdm import BABEL_SingleEval, BABEL_TransitionsEval


MOTION_TYPES = [
    '_0',
    '_1',
    '_0_with_transition',
    '_1_with_transition',
]

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def process_tokens(tokens, max_text_len, w_vectorizer):
    if len(tokens) < max_text_len:
        # pad with "unk"
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
        tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
    else:
        # crop
        tokens = tokens[:max_text_len]
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)

    pos_one_hots = []
    word_embeddings = []
    for token in tokens:
        word_emb, pos_oh = w_vectorizer[token]
        pos_one_hots.append(pos_oh[None, :])
        word_embeddings.append(word_emb[None, :])
    pos_one_hots = np.concatenate(pos_one_hots, axis=0)
    word_embeddings = np.concatenate(word_embeddings, axis=0)
    return word_embeddings, pos_one_hots, sent_len, '_'.join(tokens)


'''For use of training text motion matching model, and evaluations'''
class HumanML3D_Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, num_frames, size=None, **kwargs):
        self.opt = opt # 데이터셋 설정 옵션 (경로, 차원 등)
        self.mean = mean # 모션 데이터의 평균값 (정규화에 사용)
        self.std = std # 모션 데이터의 표준편차 (정규화에 사용)
        self.w_vectorizer = w_vectorizer # 단어를 벡터로 변환하는 GloVe 벡터라이저
        self.max_length = 20 # 텍스트 시퀀스 최대 길이
        self.pointer = 0  # 데이터 인덱스 포인터 (짧은 시퀀스 필터링용)

        # start loading dataset
        self.num_frames = num_frames if num_frames else False
        self.max_motion_length = opt.max_motion_length # 196
        # num_frames: Motion 시퀀스 길이 제약 조건. False면 제약 없음, int면 고정 길이, (min, max)면 범위 지정 (가변 길이)
        if (self.num_frames == False) or type(self.num_frames)==int:
            self.min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        else:
            self.min_motion_len = self.num_frames[0]
            self.max_motion_length = self.num_frames[1]

        # 최소 모션 길이 설정. HumanML3D는 40, KIT-ML은 24
        self.precomputed_folder = "./dataset/HumanML3D/tmp/"
        os.makedirs(self.precomputed_folder, exist_ok=True)
        # 캐시 폴더 생성. 전처리된 데이터를 저장할 디렉토리
        suffix = f"{num_frames}" if num_frames == False or type(num_frames)==int else f"{num_frames[0]}_{num_frames[1]}"
        self.split = split_file.split('/')[-1].split('.')[0]  # split_file = './dataset/HumanML3D/train, val.txt'
        
        # 데이터 로딩 분기. 조건: 캐시 파일이 없으면 새로 로딩, 있으면 캐시에서 읽기
        
        # 캐시가 없을 때: 데이터셋 로딩 및 전처리
        if not os.path.exists(self.precomputed_folder) or not os.path.exists(os.path.join(self.precomputed_folder, f'{self.split}_data_{suffix}.pkl')):
            
            # CLIP 모델 로딩. 텍스트 임베딩 생성용 (학습 시에만 사용)
            from data_loaders.amass.babel_flowmdm import load_and_freeze_clip, encode_text
            clip_model = load_and_freeze_clip('ViT-B/32').to('cuda')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            data_dict = {}
            id_list = []
    
            # split 파일 읽기: train.txt에서 파일 이름 리스트 가져오기
            with cs.open(split_file, 'r') as f: # './dataset/HumanML3D/train.txt'
                for line in f.readlines():
                    id_list.append(line.strip()) # ['M000000', 'M000001', ...]
            id_list = id_list[:size] # size로 제한 (None이면 전체)

            # 필터링된 데이터 리스트
            new_name_list = [] # 실제 사용할 샘플 이름들
            length_list = [] # 각 샘플의 길이
            
            # 각 파일 반복 처리
            for name in tqdm(id_list): # 'M000000', 'M000001', ...
                try:
                    # Motion 파일 로딩: .npy 파일에서 numpy 배열 읽기
                    motion = np.load(pjoin(opt.motion_dir, name + '.npy')) # = './dataset/HumanML3D/new_joint_vecs/M000000.npy' # shape: [num_frames, 263]
                    
                    # (추가) Angular Velocity 파일 로딩
                    angular_velocity = np.load(pjoin(opt.angular_velocity_dir, name + '.npy')) # = './dataset/HumanML3D/new_angular_velocity/M000000.npy' # shape: [num_frames, 21, 3]
                    #print(angular_velocity)
                    
                    # 필터링 조건. 너무 짧은 시퀀스 (<40 프레임) 또는 너무 긴 시퀀스 (>=200 프레임)는 제거
                    if (len(motion)) < self.min_motion_len or (len(motion) >= 200):
                        continue
                    
                    text_data = [] # 이 motion에 대한 모든 텍스트 설명들
                    flag = False # 전체 시퀀스 사용 여부
                    
                    # 텍스트 파일 로딩
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f: # './dataset/HumanML3D/texts/M000000.txt'
                        
                        # 텍스트 설명 파싱
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            
                            caption = line_split[0] # 자연어 설명. "a person walks forward"
                            tokens = line_split[1].split(' ') # 토큰화된 단어 리스트. ["a", "person", "walk", "forward"]
                            f_tag = float(line_split[2]) # 시작 시간 (초). 0.0이면 전체 시퀀스
                            to_tag = float(line_split[3]) # 끝 시간 (초). 0.0이면 전체 시퀀스
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            # 텍스트 임베딩: 학습 시에만 CLIP으로 인코딩
                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if self.split == 'train':
                                text_dict['text_embedding'] = encode_text(clip_model, [caption], device).cpu().numpy()
                            
                            # 전체 시퀀스 설명: 시간 구간이 없으면 (0.0) 전체 motion 사용
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True # 전체 시퀀스 사용
                                text_data.append(text_dict)
                            
                            # 부분 시퀀스 추출. f_tag=1.5, to_tag=3.0이면 1.5초~3.0초 구간 사용. motion[30:60] (30=1.5*20, 60=3.0*20)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    n_angular_velocity = angular_velocity[int(f_tag*20) : int(to_tag*20)]  # 추가!
        
                                    if (len(n_motion)) < self.min_motion_len or (len(n_motion) >= 200): # 추출된 부분 시퀀스도 길이 조건 체크
                                        continue
                                    
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    
                                    # num_frames 제약이 있는 경우
                                    if self.num_frames != False:
                                        # 길이가 max보다 긴 경우: 랜덤 위치에서 max_motion_length=196 만큼 크롭
                                        if len(n_motion) >= self.max_motion_length:
                                            bias = random.randint(0, len(n_motion) - self.max_motion_length)
                                            data_dict[new_name] = {'motion': n_motion[bias: bias+self.max_motion_length],
                                                                'angular_velocity': n_angular_velocity[bias: bias+self.max_motion_length],  # 추가! 랜덤 위치에서 196프레임만 자르기
                                                                'length': self.max_motion_length,
                                                                'text': [text_dict]}
                                            length_list.append(self.max_motion_length)
                                        # 길이가 max보다 짧은 경우: 그대로 저장 (나중에 패딩할 예정)
                                        else:
                                            data_dict[new_name] = {'motion': n_motion,
                                                                'angular_velocity': n_angular_velocity,  # 추가! 그대로 저장
                                                                'length': len(n_motion),
                                                                'text': [text_dict]}
                                            length_list.append(len(n_motion))

                                    # num_frames 제약이 없는 경우
                                    else:
                                        # 그대로 저장
                                        data_dict[new_name] = {'motion': n_motion,
                                                            'angular_velocity': n_angular_velocity,  # 추가! 그대로 저장
                                                            'length': len(n_motion),
                                                            'text':[text_dict]}
                                        length_list.append(len(n_motion))

                                    new_name_list.append(new_name)
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break

                    # 전체 시퀀스를 사용하는 경우. 전체 motion과 모든 텍스트 설명 저장
                    if flag:
                        if self.num_frames != False:
                            if len(motion) >= self.max_motion_length:
                                bias = random.randint(0, len(motion) - self.max_motion_length)
                                data_dict[name] = {'motion': motion[bias: bias + self.max_motion_length],
                                                    'angular_velocity': angular_velocity[bias: bias + self.max_motion_length],  # 추가! 랜덤 위치에서 196프레임만 자르기
                                                    'length': self.max_motion_length,
                                                    'text': [text_dict]}
                                length_list.append(self.max_motion_length)

                            else:
                                data_dict[name] = {'motion': motion,
                                                'angular_velocity': angular_velocity,  # 추가! 그대로 저장
                                                'length': len(motion),
                                                'text': text_data}
                                length_list.append(len(motion))

                        else:
                            data_dict[name] = {'motion': motion,
                                            'angular_velocity': angular_velocity,  # 추가! 그대로 저장
                                            'length': len(motion),
                                            'text': text_data}
                            length_list.append(len(motion))

                        new_name_list.append(name)
                except Exception as e:
                    print(e)
                    pass

            # 데이터 정렬 및 저장
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1])) # 길이 순으로 정렬: 짧은 시퀀스부터 긴 시퀀스 순서

            # 인스턴스 변수에 저장
            self.length_arr = np.array(length_list)
            self.data_dict = data_dict
            self.name_list = name_list

            # 캐시 파일 저장. ./dataset/HumanML3D/tmp/train_data_False.pkl
            data_to_store = {'data_dict': data_dict, 'name_list': name_list, 'length_list': length_list}
            with open(os.path.join(self.precomputed_folder, f'{self.split}_data_{suffix}.pkl'), 'wb') as f:
                pickle.dump(data_to_store, f)
        
        # 캐시가 있을 때: 캐시에서 로딩
        else: 
            # 빠른 로딩: 두 번째 실행부터는 pickle 파일에서 즉시 로드
            with open(os.path.join(self.precomputed_folder, f'{self.split}_data_{suffix}.pkl'), 'rb') as f:
                data_to_store = pickle.load(f)
                self.data_dict = data_to_store['data_dict']     
                self.name_list = data_to_store['name_list']
                self.length_arr = np.array(data_to_store['length_list'])


                #data_to_store = {'data_dict': data_dict, 'name_list': name_list, 'length_list': length_list}
                #with open(os.path.join(self.precomputed_folder, f'{split}_data_{suffix}.pkl'), 'wb') as f:
                #    pickle.dump(data_to_store, f)
        
        self.reset_max_len(self.max_length) # 최대 길이 설정 (20)

    # 포인터 설정: length보다 긴 시퀀스들의 시작 인덱스 찾기. length=20이면 길이 20 이상인 샘플들만 사용
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    # 유효한 샘플 개수 반환
    def __len__(self):
        return len(self.data_dict) - self.pointer
    
    def process_tokens(self, tokens):
        return process_tokens(tokens, self.opt.max_text_len, self.w_vectorizer)

    # 배치 샘플링 시 호출. item=0이면 pointer 이후 첫 번째 샘플
    def __getitem__(self, item):
        #print(self.pointer, item)
        idx = self.pointer + item
        
        # 캐시된 data_dict에서 가져오기
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        angular_velocity = data['angular_velocity']  # 추가!
        
        # Randomly select a caption. (랜덤 캡션 선택: 한 motion에 여러 설명이 있으면 하나 랜덤 선택)
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        text_embedding = text_data['text_embedding'].squeeze() if self.split == "train" else []

        # GloVe 임베딩 + POS 태그 생성
        word_embeddings, pos_one_hots, sent_len, tokens = self.process_tokens(tokens)

        m_length = max(m_length, self.min_motion_len)
        
        # 랜덤 크롭: 매 에폭마다 다른 위치에서 크롭 (Data Augmentation)
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]
        
        
        
        
        
        # Angular Velocity 처리: motion보다 1프레임 짧을 수 있음
        if len(angular_velocity) == len(data['motion']) - 1:
            # T-1 길이인 경우: 마지막 프레임 복사
            angular_velocity_crop = angular_velocity[idx:min(idx+m_length-1, len(angular_velocity))]
            # 길이 맞추기
            if len(angular_velocity_crop) < m_length:
                padding_frames = m_length - len(angular_velocity_crop)
                # 마지막 프레임 복사 (관성 유지)
                last_frame = angular_velocity_crop[-1:] if len(angular_velocity_crop) > 0 else np.zeros((1, 21, 3))
                angular_velocity_crop = np.concatenate([
                    angular_velocity_crop,
                    np.tile(last_frame, (padding_frames, 1, 1))
                ], axis=0)
        else:
            # 이미 길이가 맞는 경우
            angular_velocity_crop = angular_velocity[idx:idx+m_length]
        
        angular_velocity = angular_velocity_crop
    
    
    
    

        # Z-normalization: 평균 0, 표준편차 1로 정규화
        motion = (motion - self.mean) / self.std

        # Zero Padding: 196프레임까지 패딩
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
            
            angular_velocity = np.concatenate([angular_velocity,
                                 np.zeros((self.max_motion_length - m_length, angular_velocity.shape[1], angular_velocity.shape[2]))
                                 ], axis=0)
    
        # Flatten
        angular_velocity = angular_velocity.reshape(angular_velocity.shape[0], -1)
            
        #print(angular_velocity.shape)
        
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens, [], text_embedding, angular_velocity # last [] is for transition sequence, which is not used in this dataset

        '''
            word_embeddings: GloVe 벡터
            pos_one_hots: POS 태그 원핫
            caption: 원본 텍스트
            sent_len: 문장 길이
            motion: 정규화된 motion [196, 263]
            m_length: 실제 길이
            tokens: 토큰 문자열
            []:
            text_embedding: CLIP 임베딩
            angular_velocity: 정규화된 각속도 [196, 21, 3]
        '''

# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, load_mode, datapath='./dataset/humanml_opt.txt', split="train", **kwargs):
        self.load_mode = load_mode
        
        self.dataset_name = 't2m'
        self.dataname = 't2m'
        self.split = split

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        
        ### 1. opt 설정 읽기 ###
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.angular_velocity_dir = pjoin(abs_base_path, opt.angular_velocity_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        opt.load_mode = load_mode
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        ### 2. mean, std 로딩. (self.mean and self.std used by the getter function) ###
        if load_mode == 'gt': # GT is always used to eval GT --> from ORIGINAL UNNORMALIZED to evaluators normalization
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif load_mode in ['train', 'eval', 'gen']: # from ORIGINAL UNNORMALIZED to training normalization
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
        self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')

        ### 3. 실제 데이터셋 생성 ###
        self.t2m_dataset = HumanML3D_Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, **kwargs)

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'


    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)
        
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
class BABEL_eval(data.Dataset):
    def __init__(self, load_mode, datapath, transforms, sampler, opt, split="train", **kwargs):
        self.load_mode = load_mode

        self.split = split
        self.datapath = datapath
        abs_base_path = f'.'

        if opt is None:
            self.opt_path = './dataset/humanml_opt.txt'
            # Configurations of T2M dataset and KIT dataset is almost the same
            dataset_opt_path = pjoin(abs_base_path, self.opt_path)
            device = None  # torch.device('cuda:4') # This param is not in use in this context
            opt = get_opt(dataset_opt_path, device)
            opt.data_root = pjoin('dataset', 'babel')
            opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
            opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
            opt.text_dir = pjoin(abs_base_path, opt.text_dir)
            opt.model_dir = None
            opt.checkpoints_dir = '.'
            opt.data_root = pjoin(abs_base_path, opt.data_root)
            opt.save_root = pjoin(abs_base_path, opt.save_root)
            opt.meta_dir = './dataset'
            opt.dim_pose = 135
            opt.foot_contact_entries = 0
            opt.dataset_name = 'babel'
            opt.decomp_name = 'Decomp_SP001_SM001_H512_babel_2700epoch'
            opt.meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'motion1', 'meta')
            opt.min_motion_length = sampler.min_len # must be at least window size
            opt.max_motion_length = sampler.max_len
        self.opt = opt

        print('Loading dataset %s ...' % opt.dataset_name)

        self.dataset_name = opt.dataset_name
        self.dataname = opt.dataset_name
        self.sampler = sampler
        self.transforms = transforms

        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
        if "transitions" in load_mode:
            self.t2m_dataset = BABEL_TransitionsEval(
                split=self.split,
                datapath=self.datapath,
                transforms=self.transforms,
                opt=self.opt,
                w_vectorizer=self.w_vectorizer, sampler=self.sampler,
                cropping_sampler=kwargs.get('cropping_sampler', False)
            )
        else:
            self.t2m_dataset = BABEL_SingleEval(
                split=self.split,
                datapath=self.datapath,
                transforms=self.transforms,
                opt=self.opt,
                w_vectorizer=self.w_vectorizer, sampler=self.sampler,
                cropping_sampler=kwargs.get('cropping_sampler', False)
            )

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def inv_transform(self, data):
        return data

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()
    

# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):
    def __init__(self, load_mode, datapath='./dataset/kit_opt.txt', split="train", **kwargs):
        super(KIT, self).__init__(load_mode, datapath, split, **kwargs)

