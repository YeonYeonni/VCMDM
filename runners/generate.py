# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import load_model
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion_mix
import json
from diffusion.diffusion_wrappers import DiffusionWrapper_FlowMDM as DiffusionWrapper
from angular_velocity.loss import compute_angular_velocity
from angular_velocity.plot import save_angular_velocity_plot
    

datasets_fps = {
    "humanml": 20,
    "babel": 30
}

# 모델이 출력한 feature vector를 3D 관절 좌표로 변환
def feats_to_xyz(sample, dataset, batch_size=1):
    
    # HumanML3D
    if dataset == 'humanml':
        n_joints = 22
        mean = np.load('dataset/HML_Mean_Gen.npy')
        std = np.load('dataset/HML_Std_Gen.npy')
        sample = sample.cpu().permute(0, 2, 3, 1)
        sample = (sample * std + mean).float() # 평균, 표준편차로 정규화 해제
        sample = recover_from_ric(sample, n_joints) # --> [1, 1, seqlen, njoints, 3] # rotation invariant coordinate에서 복원
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) # --> [1, njoints, 3, seqlen]
        
    # Babel
    elif dataset == 'babel': # [bs, 135, 1, seq_len] --> 6 * 22 + 3 for trajectory
        from data_loaders.amass.transforms import SlimSMPLTransform
        transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True)
        all_feature = sample #[bs, nfeats, 1, seq_len]
        all_feature_squeeze = all_feature.squeeze(2) #[bs, nfeats, seq_len]
        all_feature_permutes = all_feature_squeeze.permute(0, 2, 1) #[bs, seq_len, nfeats]
        splitted = torch.split(all_feature_permutes, all_feature.shape[0]) #[list of [seq_len,nfeats]]
        sample_list = []
        for seq in splitted[0]:
            all_features = seq
            Datastruct = transform.SlimDatastruct
            datastruct = Datastruct(features=all_features)
            sample = datastruct.joints

            sample_list.append(sample.permute(1, 2, 0).unsqueeze(0))
        sample = torch.cat(sample_list)
        
    else:
        raise NotImplementedError("'feats_to_xyz' not implemented for this dataset")
    return sample

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = datasets_fps[args.dataset]
    assert args.instructions_file == '' or 'json' == args.instructions_file.split('.')[-1], "Instructions file must be a json file"
    dist_util.setup_dist(args.device)
    if out_path == '': # if unspecified, save in the same folder as the model
        out_path = os.path.join(os.path.dirname(args.model_path),
                                '{}_s{}'.format(niter, args.seed))
        if args.instructions_file != '':
            out_path += '_' + os.path.basename(args.instructions_file).replace('.json', '').replace(' ', '_').replace('.', '')

    animation_out_path = out_path
    os.makedirs(animation_out_path, exist_ok=True)

    # ================= Load texts + lengths and adapt batch size ================
    # this block must be called BEFORE the dataset is loaded
    is_using_data = args.instructions_file == ''
    
    # JSON 파일이 있으면
    if not is_using_data:
        assert os.path.exists(args.instructions_file)
        # load json
        with open(args.instructions_file, 'r') as f:
            instructions = json.load(f)
            
            assert "text" in instructions and "lengths" in instructions, "Instructions file must contain 'text' and 'lengths' keys"
            assert len(instructions["text"]) == len(instructions["lengths"]), "Instructions file must contain the same number of 'text' and 'lengths' elements"
        
        num_instructions = len(instructions["text"])
        args.batch_size = num_instructions
        args.num_samples = 1
    else:
        num_instructions = args.num_samples
        args.batch_size = num_instructions
        args.num_samples = 1

    # ================= Load dataset or prepare model_kwargs for inference ================
    if is_using_data:
        print('Loading dataset...')
        if args.split == "test" and args.dataset == "babel":
            args.split = "val" # Babel does not have a test set

        try:
            data = load_dataset(args, args.split)
        except Exception as e:
            print(f'Error while loading dataset: {e}')
            return
        
        if is_using_data:
            iterator = iter(data)
            sample_gt, model_kwargs = next(iterator) # 한 배치 가져오기
            
        # 사용할 텍스트를 JSON으로 저장
        j = { "sequence": [] }
        for i in range(num_instructions):
            length = model_kwargs['y']['lengths'][i].item()
            text = model_kwargs['y']['text'][i]
            j["sequence"].append([length, text])
        with open(os.path.join(animation_out_path, "prompted_texts.json"), "w") as f:
            json.dump(j, f)

        # CFG 설정
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
            
    # JSON 파일 사용
    else:
        json_lengths = instructions["lengths"]
        json_texts = instructions["text"]
        json_angular_velocity = instructions["angular_velocity"] if "angular_velocity" in instructions else None
        #print(json_angular_velocity)
        
        # 마스크 생성: 실제 모션 길이만 1, 나머지는 0
        mask = torch.ones((len(json_texts), max(json_lengths)))
        for i, length in enumerate(json_lengths):
            mask[i, length:] = 0 # 길이 이후는 패딩
            
            
            
        # angular_velocity 텐서 생성 [동작개수, 최대길이, 8]
        angular_velocity_tensor = torch.zeros((len(json_texts), max(json_lengths), 8))
        if json_angular_velocity:
            for i in range(len(json_texts)):
                for j in range(8):
                    angular_velocity_tensor[i, :json_lengths[i], j] = json_angular_velocity[i][j]
        
        
            
        # 모델에 전달할 조건 딕셔너리
        model_kwargs = {'y': {
            'mask': mask,
            'lengths': torch.tensor(json_lengths),
            'text': list(json_texts),
            'angular_velocity': angular_velocity_tensor,
            'tokens': [''],
        }}
        
        with open(os.path.join(animation_out_path, "prompted_texts.json"), "w") as f:
            json.dump(instructions, f)
            
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        print(list(zip(list(model_kwargs['y']['text']), list(model_kwargs['y']['lengths'].cpu().numpy()))))

    # ================= Load model and diffusion wrapper ================
    print("Creating model and diffusion...")
    model, diffusion = load_model(args, dist_util.dev()) # ckpt에서 로드
    diffusion = DiffusionWrapper(args, diffusion, model) # FlowMDM wrapper

    # ================= Sample ================
    all_motions = []
    all_lengths = []
    all_text = []
    all_angular_vel = []
    all_samples_raw = []  # 원본 sample 저장용
    
    for rep_i in range(args.num_repetitions): # num_repetitions 만큼 반복 생성
        print(f'### Sampling [repetition #{rep_i}]')
        
         # 노이즈에서 시작해 점진적으로 모션 생성 (denoising process)
        sample = diffusion.p_sample_loop(
            clip_denoised=False,
            model_kwargs=model_kwargs, # 텍스트, 길이, 마스크 포함
            progress=True,
        )
        # sample 형태: [batch, features, 1, seq_len]
        
        all_samples_raw.append(sample.cpu())  # 원본 저장 (xyz 변환 전)
    
        sample = feats_to_xyz(sample, args.dataset) # 생성한 특징 벡터를 3D 관절 좌표로 변환 [batch, joints, 3, seq_len]

        # 모든 텍스트를 "///"로 연결하여 저장
        c_text = ""
        for i in range(num_instructions):
            c_text += model_kwargs['y']['text'][i] + " /// "

        all_text.append(c_text)
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].sum().unsqueeze(0))#.cpu().numpy())
        all_angular_vel.append(model_kwargs['y']['angular_velocity'])

        print(f"created {rep_i+1}/{args.num_repetitions} human motion compositions.")

    all_motions = np.concatenate(all_motions, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)
    all_angular_vel = np.concatenate(all_angular_vel, axis=0)

    # ================= Save results + visualizations 결과 저장 ================
    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    
    np.save(npy_path,
            {'motion': all_motions, 
             'text': all_text, 
             'lengths': all_lengths,
             'angular_velocity': all_angular_vel,
             'num_samples': args.num_samples, 
             'num_repetitions': args.num_repetitions
             })
    
    # 텍스트만 따로 저장
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
        
    # 길이만 따로 저장
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))
        
    # 각속도만 따로 저장
    with open(npy_path.replace('.npy', '_ang_vel.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_angular_vel]))

    # 시각화 생성
    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.t2m_kinematic_chain # 관절 연결 관계
    
    sample_print_template, row_print_template, \
    sample_file_template, row_file_template = construct_template_variables(args.unconstrained)

    try:
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.num_samples] # 'walk /// sit /// ...'
            motion = all_motions[rep_i*args.num_samples].transpose(2, 0, 1) # [joints, 3, seq_len] → [seq_len, joints, 3]
            
            save_file = sample_file_template.format(rep_i) # "sample_rep00.mp4"
            print(sample_print_template.format(rep_i, save_file))
            animation_save_path = os.path.join(animation_out_path, save_file)
            
            lengths_list = model_kwargs['y']['lengths'] # [40, 30]
            
            # 각 프레임에 캡션 할당. 예: ["walk"]*40 + ["sit"]*30
            captions_list = []
            for c, l in zip(caption.split(" /// "), lengths_list):
                captions_list += [c,] * l
                
            # 시각화 생성
            plot_3d_motion_mix(animation_save_path, skeleton, motion, dataset=args.dataset, title=captions_list, fps=fps,
                        vis_mode='alternate', lengths=lengths_list)
            
            
            
            # Angular Velocity 플롯 생성
            try:
                # 원본 sample에서 angular velocity 계산
                sample_raw = all_samples_raw[rep_i].to(dist_util.dev())  # [1, 263, 1, nframes]
                angular_vel = compute_angular_velocity(sample_raw, dataset=args.dataset, fps=fps)  # [1, 8, 1, nframes-1]
                
                # model_kwargs 준비 (text 정보)
                plot_kwargs = {'y': {'text': [caption]}}
                
                # 플롯 저장
                save_angular_velocity_plot(angular_vel, animation_out_path, rep_i, plot_kwargs)
            except Exception as e:
                print(f'  Warning: Could not generate angular velocity plot: {e}')
            
            
            
            
            
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)
    except Exception as e:
        print(f'Error while processing sample: {e}')

    # 여러 샘플 결합
    save_multiple_samples(args, animation_out_path,
                                            row_print_template, row_file_template,
                                            caption, rep_files)

    abs_path = os.path.abspath(animation_out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, row_file_template, caption, rep_files):
    all_rep_save_file = row_file_template # "sample_all.mp4"
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files] # ffmpeg 입력 파일들. [' -i sample_rep00.mp4 ', ' -i sample_rep01.mp4 ', ...]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else '' # hstack: 수평으로 나란히 배치
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}' # ffmpeg 명령 실행
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(all_rep_save_file))


def construct_template_variables(unconstrained):
    row_file_template = 'sample_all.mp4'
    if unconstrained:
        sample_file_template = 'sample_rep{:02d}.mp4'
        sample_print_template = '[rep #{:02d} | -> {}]'
        row_print_template = '[all repetitions | -> {}]'
    else:
        sample_file_template = 'sample_rep{:02d}.mp4'
        sample_print_template = '[Rep #{:02d} | -> {}]'
        row_print_template = '[all repetitions | -> {}]'

    return sample_print_template, row_print_template, \
           sample_file_template, row_file_template


def load_dataset(args, split):
    n_frames = 150 # this comes from PriorMDM, so I'm using it here as well
    if args.dataset == 'babel':
        args.num_frames = (args.min_seq_len, args.max_seq_len) # 가변 길이
    else:
        args.num_frames = n_frames

    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=args.num_frames,
                              split=split,#split,
                              load_mode='gen',#'eval',
                              protocol=args.protocol,
                              pose_rep=args.pose_rep,
                              num_workers=1)
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
