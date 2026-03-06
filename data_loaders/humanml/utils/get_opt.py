import os
from argparse import Namespace
import re
from os.path import join as pjoin
from data_loaders.humanml.utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device): # opt_path = './dataset/humanml_opt.txt'
    opt = Namespace() # 빈 객체 생성
    opt_dict = vars(opt) # 딕셔너리로 변환

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    
    with open(opt_path) as f:
        for line in f: # 예: "dataset_name: t2m"
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ') # key = "dataset_name", value = "t2m"
                
                # 타입 변환
                if value in ('True', 'False'):
                    opt_dict[key] = bool(value)
                elif is_float(value): # "3.14" → 3.14
                    opt_dict[key] = float(value)
                elif is_number(value): # "196" → 196
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value) # "t2m" → "t2m"

    # print(opt)
    opt_dict['which_epoch'] = 'latest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    ## 가장 중요한 부분: 데이터셋별 경로 및 파라미터 설정 ##
    # 1. HumanML3D 데이터셋
    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs') # = './dataset/HumanML3D/new_joint_vecs'
        opt.angular_velocity_dir = pjoin(opt.data_root, 'new_angular_velocity')  # = './dataset/HumanML3D/new_angular_velocity'
        opt.text_dir = pjoin(opt.data_root, 'texts') # = './dataset/HumanML3D/texts'
        opt.joints_num = 22
        opt.dim_pose = 263 # 모션 데이터의 차원
        opt.max_motion_length = 196
        
    # 2. KIT-ML 데이터셋
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.angular_velocity_dir = pjoin(opt.data_root, 'new_angular_velocity')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
        
    else:
        raise KeyError('Dataset not recognized')

    ## 추가 설정
    opt.dim_word = 300 # GloVe 워드 임베딩 차원
    opt.num_classes = 200 // opt.unit_length # 클래스 수
    opt.dim_pos_ohot = len(POS_enumerator) # POS 태그 개수
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt