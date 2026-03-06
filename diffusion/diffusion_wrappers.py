
import torch
import torch as th
from copy import deepcopy

'''
여러 동작을 하나의 시퀀스로 결합하고 Reverse Diffusion 수행

1. 클래스 구조 개요
2. init() - 초기화
3. add_bias_and_absolute_matrices() - Attention 제어
4. add_conditions_mask() - 동작 마스크
5. p_sample_loop() - 샘플링 래퍼
6. p_sample_loop_progressive() - 핵심 샘플링
7. Angular Velocity 재구성
8. 전체 데이터 흐름

class DiffusionWrapper_FlowMDM():
    ├── __init__()                       # 초기화
    ├── add_bias_and_absolute_matrices() # Self-Attention 제어 행렬 생성
    ├── add_conditions_mask()            # 동작별 시간 마스크
    ├── p_sample_loop()                  # 전체 샘플링 (최종 결과만)
    └── p_sample_loop_progressive()      # 점진적 디노이징 (중간 과정 포함)
'''

class DiffusionWrapper_FlowMDM():
    def __init__(self, args, diffusion, model):
        self.model = model # FlowMDM 모델
        self.diffusion = diffusion # Gaussian Diffusion 객체 (노이즈 스케줄링. 디노이징 스텝 제공)
        self.guidance_param = args.guidance_param # CFG 강도 조절 파라미터

    # Attention 제어 행렬 생성. 여러 동작을 결합할 때 각 동작이 독립적인 positional encoding을 갖도록 하고, 같은 동작끼리만 attention 하도록
    def add_bias_and_absolute_matrices(self, model_kwargs, shape, device):
        """
        We build:
        > pe_bias --> [T, T] matrix with -inf and 0's limiting where the attention during APE mode focuses (0's), i.e., inside each subsequence
        > pos_pe_abs --> [T] matrix with the absolute position of each frame in each subsequence (for injecting the APE sinusoidal correctly during APE mode).
        
        생성하는 것:
        > pe_bias: [T, T] - Self-Attention 제한 행렬 (-inf와 0)
        > pos_pe_abs: [T] - 각 동작 내 절대 위치 (0부터 시작)
        """
        nframes = shape[-1] # 전체 프레임 수

        # Step 1: 행렬 초기화
        pos_pe_abs = torch.zeros((nframes, ), device=device, dtype=torch.float32) # 위치 인코딩용 1D 벡터 [179]
        pe_bias = torch.full((nframes, nframes), float('-inf'), device=device, dtype=torch.float32) # Attention bias 행렬 [179, 179] - 기본값은 -inf (attention 불가)

        # Step 2: 각 동작별로 PE 및 Attention 제한 설정
        s = 0 # 시작 위치 (start)
        for length in model_kwargs['y']['lengths']:
            pos_pe_abs[s:s+length] = torch.arange(length, device=device, dtype=torch.float32)
            pe_bias[s:s+length, s:s+length] = 0 # only attend to the segment for the absolute modeling part of the schedule
            s += length
        '''
            (예시)
            [반복 1 (length=120, kick):]
                pos_pe_abs[s:s+length] = torch.arange(length, device=device, dtype=torch.float32)
                # pos_pe_abs[0:120] = [0, 1, 2, ..., 119]
                
                pe_bias[s:s+length, s:s+length] = 0
                # pe_bias[0:120, 0:120] = 0 (120x120 블록)
                # → "kick" 동작끼리만 attend 가능
                
                s += length  # s = 120

            [반복 2 (length=59, walk):]
                pos_pe_abs[120:179] = [0, 1, 2, ..., 58]
                # 다시 0부터 시작! (독립적인 위치)
                
                pe_bias[120:179, 120:179] = 0
                # → "walk" 동작끼리만 attend 가능
                
                s += length  # s = 179
        '''

        '''
            (최종 결과)
            pos_pe_abs (각 동작마다 0부터 시작):
                [0, 1, 2, ..., 119, 0, 1, 2, ..., 58]
                ← kick (120)    →  ←  walk (59)  →

            pe_bias (블록 대각 행렬):
                0...119  120...178
                0...119 [  0       -∞   ]  ← kick는 kick끼리만
                120..178[ -∞        0   ]  ← walk는 walk끼리만

        '''

        model_kwargs['y']['pe_bias'] = pe_bias # in FlowMDM forward, it is selected according to the BPE schedule if active
        model_kwargs['y']['pos_pe_abs'] = pos_pe_abs.unsqueeze(0) # needs batch size

    # 동작 마스크. 각 프레임이 어느 동작에 속하는지 표시하는 boolean 마스크 생성
    def add_conditions_mask(self, model_kwargs, num_frames, device):
        """
        We build a mask of shape [S, T, 1] where S is the number of motion subsequences, T is the max. sequence length.
        For each subsequence, the mask is True only for the frames that belong to the subsequence.
        
        Shape: [S, T, 1]
            - S: 동작 개수
            - T: 전체 프레임
            - 1: Feature dimension (broadcasting용)
        """
        num_samples = len(model_kwargs["y"]["lengths"]) # 동작 개수 S

        # Step 1: 마스크 초기화
        conditions_mask = th.zeros((num_samples, num_frames, 1), device=device, dtype=th.bool) # [2, 179, 1] - 모두 False (0)
        
        # Step 2: 각 동작별로 마스크 채우기
        s = 0 # 시작 위치
        MARGIN = 0 # 여백 (현재 사용 안함)
        for i, length in enumerate(model_kwargs["y"]["lengths"]):
            conditions_mask[i, s+MARGIN:s+length-MARGIN, :] = True # all batch elements have the same instructions
            s += length
        '''
            (예시)
            [반복 1 (i=0, length=120):]
                conditions_mask[i, s+MARGIN:s+length-MARGIN, :] = True
                # conditions_mask[0, 0:120, :] = True
                # → 동작 0의 구간(0~119)만 True
                
                s += length  # s = 120

            [반복 2 (i=1, length=59):]
                conditions_mask[1, 120:179, :] = True
                # → 동작 1의 구간(120~178)만 True   
                
                s += length  # s = 179
        '''

        '''
            (최종 결과 예시 )
            conditions_mask[0, :, 0] = [T,T,...,T(120개), F,F,...,F(59개)]  # kick
            conditions_mask[1, :, 0] = [F,F,...,F(120개), T,T,...,T(59개)]  # walk  
        '''

        model_kwargs['y']['conditions_mask'] = conditions_mask


    # 샘플링 래퍼. p_sample_loop_progressive() 호출하고 최종 결과만 반환
    '''
        p_sample_loop_progressive() (Generator)
            ↓ yield step 0
            ↓ yield step 1
            ↓ ...
            ↓ yield step 999 (최종)
            
        p_sample_loop()
            → final = step 999
            → return final["sample"]
    '''
    def p_sample_loop(
        self,
        model_kwargs=None, # list of dicts
        **kwargs,
    ):
        final = None
        for i, sample in enumerate(self.p_sample_loop_progressive(
            model_kwargs=model_kwargs,
            **kwargs,
        )):
            final = sample
        return final["sample"]

    # 핵심 샘플링 함수
    def p_sample_loop_progressive(
        self,
        noise=None, # # 초기 노이즈 (None이면 랜덤 생성)
        model_kwargs=None, # list of dicts. 조건 정보
        device=None, # 디바이스
        progress=False, # 진행바 표시 여부
        **kwargs,
    ):
        # Step 1: shape 설정
        bs, nframes = 1, model_kwargs['y']['lengths'].sum().item() # bs = 1 (모든 동작을 하나의 배치로), nframes = 모든 동작을 합친 전체 프레임 수
        shape = (bs, self.model.njoints, self.model.nfeats, nframes) # all batch elements form the same sequence
        # (1, 263, 1, 179)
        # - 263: Feature dimension (joints × representation)
        # - 1: 항상 1 (차원 확장용)

        # Step 2: Device 설정
        if device is None:
            device = next(self.model.parameters()).device # 모델이 있는 device (GPU/CPU) 자동 감지
        assert isinstance(shape, (tuple, list))
        
        # Step 3: 초기 노이즈 생성
        if noise is not None:
            img = noise # 외부에서 제공된 노이즈 사용
        else:
            img = th.randn(*shape, device=device) # 랜덤 가우시안 노이즈 [1, 263, 1, 179]

        # Step 4: model_kwargs 재구성
        model_kwargs = deepcopy(model_kwargs) # 원본 보존

        # 1. 동작 마스크 추가
        self.add_conditions_mask(model_kwargs, nframes, device) # → model_kwargs['y']['conditions_mask'] = [bs, nframes, 1]
        
        # 2. Attention 제어 행렬 추가
        self.add_bias_and_absolute_matrices(model_kwargs, shape, device)
        # → model_kwargs['y']['pe_bias'] = [179, 179]
        # → model_kwargs['y']['pos_pe_abs'] = [1, 179]

        ######################################
        ###### Angular velocity 재구성 #######
        ######################################
        
        # Angular velocity를 여러 동작에서 하나의 시퀀스로 통합. 통합된 [1, max_len, 8]으로 변환
        if 'angular_velocity' in model_kwargs['y']:

            # Step 1: 원본 데이터 가져오기
            angular_vel_segments = model_kwargs['y']['angular_velocity']  # [S, max_len, 8]
            # [2, 120, 8] - 동작별 분리된 상태
            # [0, :, :] = 0.9 (kick, 120 프레임)
            # [1, :, :] = 0.2 (walk, 59 프레임 + 61 padding)
            
            lengths = model_kwargs['y']['lengths']  # 원본 lengths [50, 30, 116]
            
            # Step 2: 통합된 angular velocity 생성 [1, nframes, 8]
            angular_vel_unified = th.zeros((1, nframes, 8), device=device)
            
            # Step 3: 동작별로 복사
            s = 0  # 시작 위치
            for i, length in enumerate(lengths):
                # i번째 동작의 angular velocity를 전체 시퀀스에 복사
                angular_vel_unified[0, s:s+length, :] = angular_vel_segments[i, :length, :]
                s += length
            '''
                (예시)

                [반복 1 (i=0, length=120, kick):]
                    angular_vel_unified[0, s:s+length, :] = angular_vel_segments[i, :length, :]
                    # angular_vel_unified[0, 0:120, :] = angular_vel_segments[0, :120, :]
                    # → [0, 0~119, 0~7] 위치에 0.9 복사
                    
                    s += length  # s = 120

                [반복 2 (i=1, length=59, walk):]
                    angular_vel_unified[0, 120:179, :] = angular_vel_segments[1, :59, :]
                    # → [0, 120~178, 0~7] 위치에 0.2 복사
                    
                    s += length  # s = 179
            '''

            # ===== 확인 코드 2: 재구성된 angular_velocity 출력 =====
            # print("\n" + "="*60)
            # print("🔄 [diffusion_wrappers.py] Angular Velocity 재구성 완료")
            # print("="*60)
            # print(f"원본 Shape: {angular_vel_segments.shape}")
            # print(f"통합 Shape: {angular_vel_unified.shape}")
            # print(f"전체 프레임: {nframes}")
            # print()
            
            # s = 0
            # for i, length in enumerate(lengths):
            #     segment_mean = angular_vel_unified[0, s:s+length, 0].mean().item()
            #     segment_min = angular_vel_unified[0, s:s+length, 0].min().item()
            #     segment_max = angular_vel_unified[0, s:s+length, 0].max().item()
                
            #     print(f"구간 {i} (프레임 {s}~{s+length-1}):")
            #     print(f"  - 평균값: {segment_mean:.4f}")
            #     print(f"  - 최소값: {segment_min:.4f}")
            #     print(f"  - 최대값: {segment_max:.4f}")
            #     print(f"  - 첫 5개 값: {angular_vel_unified[0, s:s+5, 0].cpu().numpy()}")
            #     s += length
            # print("="*60 + "\n")
            # ===== 확인 코드 2 끝 =====
            
            # Step 4: 저장
            model_kwargs['y']['all_angular_velocity'] = angular_vel_segments # 각 동작 원본 보존
            model_kwargs['y']['angular_velocity'] = angular_vel_unified # 통합된 새 버전 저장
            '''
                변환 전 (분리):
                angular_vel_segments[0] = [0.9] x 120  ─┐
                angular_vel_segments[1] = [0.2] x 59   ─┤
                                                        │
                                                        ↓ 재구성
                변환 후 (통합):
                angular_vel_unified[0] = [0.9 x 120 | 0.2 x 59]
                                        0───119   120─178
            '''
        
        ######################################
        ## End of angular velocity 재구성 ####
        ######################################

        # 메타데이터 통합
        model_kwargs["y"]["mask"] = th.ones((bs, nframes), device=device, dtype=th.bool)
        model_kwargs["y"]["lengths"] = th.tensor([nframes], device=device, dtype=th.int64)
        model_kwargs["y"]["scale"] = th.ones(bs, device=device) * self.guidance_param
        
        # 텍스트 결합. texts are joined as well
        model_kwargs["y"]["all_texts"] = [model_kwargs["y"]["text"], ] # 개별 텍스트
        model_kwargs["y"]["all_lengths"] = [model_kwargs["y"]["lengths"], ] # 개별 길이
        model_kwargs["y"]["text"] = " -- ".join(model_kwargs["y"]["text"]) # 단일 텍스트 문자열로 통합

        # Reverse Diffusion 과정
        indices = list(range(self.diffusion.num_timesteps))[::-1] # [999, 998, 997, ..., 2, 1, 0]
        
        # Progress bar 설정
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices) # 100%|███████| 1000/1000 [00:21<00:00, 46.85it/s]

        # 디노이징 루프
        for t in indices: # t = 999, 998, ..., 0
                
            with th.no_grad(): # Gradient 계산 안 함 (inference)

                t = th.tensor([t] * shape[0], device=device) # 현재 timestep 설정. [999] → 배치 크기만큼 복사 (shape[0]=1)
                
                # 한 스텝 디노이징 수행
                out = self.diffusion.p_sample(
                    self.model, # FlowMDM 모델
                    img, # 현재 노이즈 샘플
                    t, # 현재 timestep
                    model_kwargs=model_kwargs, # 조건 정보
                    **kwargs,
                )

                # 중간 결과 반환
                yield out # Generator로 중간 과정 전달 (progressive 시각화용)
                img = out["sample"] # 다음 스텝을 위한 샘플 업데이트. 디노이징된 결과가 다음 스텝의 입력이 됨
