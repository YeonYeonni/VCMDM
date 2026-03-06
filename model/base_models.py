
import torch
import clip
import torch.nn as nn
from model.MDM import PositionalEncoding, TimestepEmbedder


class TextConditionalModel(nn.Module):
    def __init__(self, latent_dim=256, cond_mode="no_cond", cond_mask_prob=0., dropout=0.0, clip_dim=512, clip_version=None, **kargs):
        super().__init__()
        self.cond_mode = cond_mode
        assert self.cond_mode in ["no_cond", "text"]
        self.cond_mask_prob = cond_mask_prob

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        
        if cond_mode != 'no_cond':
            if 'text' in cond_mode:
                self.embed_text = nn.Linear(clip_dim, latent_dim)
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            else:
                raise NotImplementedError("only conditioning with text is implemented for now")

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    
    def compute_embedding(self, x, timesteps, y):
        """
            Explanation on what the buffers do:
            - emb: stores the embedding for the current condition. It is used to avoid recomputing the embedding if the condition is the same (big inference speedup)
            - emb_hash: stores the hash of the condition. It is used to check if the condition is the same as the one stored in emb
            - emb_forcemask: stores the embedding for the current condition, but with the mask forced to True. It is used to avoid recomputing the embedding for the unconditional case
            - emb_forcemask_hash: stores the hash of the condition. It is used to check if the condition is the same as the one stored in emb_forcemask
        """
        bs, njoints, nfeats, nframes = x.shape # [bs, 263, 1, 196]
        multitext_mode = "all_texts" in y or not isinstance(y['text'][0], str)
        key = "all_texts" if "all_texts" in y else "text"

        time_emb = self.embed_timestep(timesteps)  # [1, bs, 512]
        force_mask = y.get('uncond', False)
        
        #######################
        ##### 개발 타겟 시작 #####
        #######################



        angular_velocity = y['angular_velocity']  # [bs, nframes, 8] # dataset 구성할 때 196 프레임으로 맞춤, 8개 관절 x 1차원 = 8
        
        
        #angular_velocity = angular_velocity.mean(dim=1)  # [bs, 8]
        #vel_emb = self.angularVelEncoder(angular_velocity)
        #vel_emb = vel_emb + self.angularVelRes(vel_emb)
        #vel_emb = self.angularVelOutNorm(vel_emb)


        # [B,T,8] -> [B,8,T] -> Conv1d -> [B,T,D]
        vel_seq = self.angularVelTemporalEncoder(angular_velocity.transpose(1, 2)).transpose(1, 2)
        vel_seq = vel_seq + self.angularVelFrameProj(angular_velocity)
        vel_seq = self.angularVelOutNorm(vel_seq)

        ###################
        ## CFG 마스킹 추가 ##
        ###################
        
        ## 1. Original CFG masking code ##
        #vel_emb = self.mask_cond(vel_emb, force_mask=force_mask)  # [bs, 512]
        #vel_emb = vel_emb.unsqueeze(0)  # [1, bs, 512]


        # 2. CFG mask: 샘플 단위 마스크를 모든 프레임에 동일 적용
        if force_mask:
            vel_seq = torch.zeros_like(vel_seq)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=vel_seq.device) * self.cond_mask_prob
            ).view(bs, 1, 1)
            vel_seq = vel_seq * (1. - mask)

        vel_emb = vel_seq.permute(1, 0, 2)  # [T,B,D]

        #############################
        ### 임베딩 벡터 계산 방식 끝 ###
        #############################
        
        
        

        # Case 1: 조건부 생성 (force_mask=False)
        if not force_mask:
            # text를 기준으로 해시 생성
            if 'text' == self.cond_mode:
                primitive = frozenset(y[key]) if not multitext_mode else frozenset((frozenset(txts) for txts in y[key]))
            else:
                raise ValueError
            
            hash_value = hash(primitive)
            recompute = not hasattr(self, 'emb_hash') or self.emb_hash != hash_value # 이전에 같은 text로 계산한 적이 있는지 확인
            if not recompute: # 이전에 계산한 적이 있으면?
                return vel_emb + time_emb + self.emb # 저장된 text embedding 재사용
            
        # Case 2: 무조건부 생성 (force_mask=True, CFG용)
        else: # force_mask=True
            # x의 shape을 기준으로 해시 생성 (text 무관)
            hash_value = hash(frozenset(x.shape))
            recompute = not hasattr(self, 'emb_forcemask_hash') or self.emb_forcemask_hash != hash_value
            if not recompute: # 이전에 계산한 적이 있으면?
                return vel_emb + time_emb + self.emb_forcemask # null embedding 재사용



        ## compute embedding ##
        
        # Case A: Single Text Mode (HumanML3D)
        if not multitext_mode: # --> single text training (e.g. HumanML3D dataset) / inference # y['text'] = ['a person walks', 'a person runs', ...]
            # CLIP으로 text 인코딩
            enc_text = self.encode_text(y['text']) if "text_embeddings" not in y else y["text_embeddings"] # if precomputed --> faster # [bs, 512] - CLIP embedding
            
            # Linear로 latent_dim으로 변환
            cond_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask)) # [bs, 512] # text embedding에 CFG 마스킹 적용
            
            # 모든 프레임에 동일한 text embedding 적용
            cond_emb = cond_emb.unsqueeze(0).expand(nframes, -1, -1) # [T, N, d] # [nframes, bs, 512]
            
        # Case B: Multi-Text Mode (BABEL - 긴 시퀀스)
        else: # --> multi-text training / inference (e.g. Babel dataset) # y['all_texts'] = [['walk'], ['run', 'jump'], ...]
            if "text_embeddings" in y: # preloaded for fast training / eval
                enc_text = y["text_embeddings"] # 여러 text를 각각 인코딩. 또는 self.encode_text() # [I, N, 512] where I=instruction 개수, N=batch size
            else:
                # 'conditions_mask' has shape [I, T, N] where I is the number of different conditions, N is batch size, T is sequence length.
                # y[key] is a list of size I with each element being a list of strings of size N
                # We need to encode the text and build the embedding matrix
                texts_list = y[key]
                # homogeneize all lists to same length to stack them later
                max_len = max([len(texts) for texts in texts_list])
                for i, texts in enumerate(texts_list):
                    if len(texts) < max_len:
                        texts_list[i] = texts + [''] * (max_len - len(texts))
                enc_text = [self.encode_text(text) for text in texts_list]
                enc_text = torch.stack(enc_text, dim=1)

            I, N, d = enc_text.shape # 모든 text embedding 계산
            enc_text = enc_text.reshape(-1, enc_text.shape[-1]) # [I*N, d]
            embedded_text = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask)).reshape(I, N, d) # [I, N, d]

            # conditions_mask: [I, nframes, bs] - 어느 프레임에 어느 instruction 적용할지
            conditions_mask = y['conditions_mask'] # [I, T, N] # [I, nframes, bs]
            conditions_mask = conditions_mask.unsqueeze(-1).expand(-1, -1, -1, self.latent_dim) # [I, T, N, d]
            
            # 각 프레임마다 해당하는 instruction의 embedding을 할당
            cond_emb = torch.zeros(conditions_mask.shape[1:], device=embedded_text.device) # [T, N, d]
            for i in range(I):
                m = conditions_mask[i] # [T, N, d] # [nframes, bs, 512] (expanded)
                cond_emb = cond_emb + m * embedded_text[i].unsqueeze(0) # [T, N, d] --> [T, N, d]
                
            '''
                # BABEL 데이터 예시
                all_texts = [['walk forward'], ['turn left'], ['sit down']]
                conditions_mask = [
                    [[1,1,...,0,0,0], ...],  # walk: frame 0-30
                    [[0,0,...,1,1,0], ...],  # turn: frame 31-50  
                    [[0,0,...,0,0,1], ...],  # sit:  frame 51-60
                ]
            '''
        
        # send to buffer
        if force_mask: # 무조건부 embedding 저장
            self.register_buffer('emb_forcemask', cond_emb, persistent=False)
            self.register_buffer('emb_forcemask_hash', torch.tensor(hash(frozenset(x.shape))), persistent=False)
        else: # 조건부 embedding 저장
            self.register_buffer('emb', cond_emb, persistent=False)
            self.register_buffer('emb_hash', torch.tensor(hash(primitive)), persistent=False)

        return time_emb + vel_emb + cond_emb # [1, bs, 512] + [1, bs, 512] + [nframes, bs, 512] = [nframes, bs, 512] (broadcasting)