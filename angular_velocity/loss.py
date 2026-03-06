import torch
from data_loaders.humanml.common.quaternion import cont6d_to_matrix, qeuler, matrix_to_quat

def so3_log_map_torch(R, eps=1e-8):
    """
    SO(3) log mapping to rotation vector (PyTorch version)
    
    Args:
        R: [..., 3, 3] - Rotation matrices in SO(3)
    
    Returns:
        rotvec: [..., 3] - Rotation vectors (axis * angle) in radians
    """
    # Trace: tr(R) = R[0,0] + R[1,1] + R[2,2]
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # [...]
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.arccos(cos_theta)  # [...]
    
    # vee operator: extract axis from skew-symmetric part
    w = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)  # [..., 3]
    
    sin_theta = torch.sin(theta)
    
    # axis = w / (2 * sin(theta))
    axis = w / (2.0 * sin_theta[..., None] + eps)
    
    # rotvec = theta * axis
    rotvec = axis * theta[..., None]  # [..., 3]
    
    # Handle theta ~ 0 (small rotation)
    small = theta < 1e-5
    rotvec[small] = 0.0
    
    return rotvec

def compute_angular_velocity(motion_data, dataset=None, fps=20):
    """
    모션 데이터로부터 각속도(Angular Velocity) 계산.
    263차원의 6D 표현에서 8개의 관절에 대한 각속도 L2 norm 계산
    (left_hip, right_hip, left_knee, right_knee, left_shoulder, right_shoulder, left_elbow, right_elbow)
        
        1. 6D rotation to rotation matrix 변환
        2. Relative rotation 계산 (현재 프레임과 다음 프레임 간의 회전 차이)
        3. Log mapping을 사용하여 rotation vector 계산
        4. Angular velocity = rotation vector * fps (radian/sec)
        5. 8개 관절 선택 및 축별 L2 norm 계산
    
    Args:
        motion_data: [bs, 263, 1, nframes] - 모션 데이터 (target 또는 model_output)
        dataset: 데이터셋 객체 (호환성 유지, 사용 안함)
        fps: int - 프레임레이트 (default: 20 fps)
    
    Returns:
        angular_velocity: [bs, 8, 1, nframes-1] - 8개 관절의 각속도 L2 norm (rad/s)
    """
    
    device = motion_data.device
    bs, _, _, nframes = motion_data.shape
    
    # 1. 6D rotation 추출: [bs, 263, 1, nframes] → [bs, 126, nframes]
    rot6d = motion_data[:, 67:193, 0, :]  # [bs, 126, nframes]
    
    # 2. Reshape: [bs, 126, nframes] → [bs, nframes, 21, 6]
    rot6d = rot6d.permute(0, 2, 1).reshape(bs, nframes, 21, 6)
    
    # 3. 6D → Rotation Matrix: [bs, nframes, 21, 6] → [bs, nframes, 21, 3, 3]
    rot6d_flat = rot6d.reshape(bs * nframes * 21, 6)
    rotmat_flat = cont6d_to_matrix(rot6d_flat)  # [bs*nframes*21, 3, 3]
    rotmat = rotmat_flat.reshape(bs, nframes, 21, 3, 3)
    
    # 4. Relative Rotation: R_rel = R_next @ R_t^T
    R_t = rotmat[:, :-1]      # [bs, nframes-1, 21, 3, 3]
    R_next = rotmat[:, 1:]    # [bs, nframes-1, 21, 3, 3]
    R_rel = torch.matmul(R_next, R_t.transpose(-2, -1))  # [bs, nframes-1, 21, 3, 3]
    
    # 5. Log Mapping: SO(3) → so(3) (rotation vector phi in radians)
    phi = so3_log_map_torch(R_rel)  # [bs, nframes-1, 21, 3] rad
    
    # 6. Angular Velocity: omega = phi * fps
    omega = phi * fps  # [bs, nframes-1, 21, 3] rad/s
    
    # 7. 8개 관절 선택 및 축별 L2 norm
    # HumanML3D joint indices:
    # 0: left_hip, 1: right_hip, 3: left_knee, 4: right_knee
    # 15: left_shoulder, 16: right_shoulder, 17: left_elbow, 18: right_elbow
    
    joint_axis_config = {
        0: [0, 2],   # left_hip: wx, wz
        1: [0, 2],   # right_hip: wx, wz
        3: [0],      # left_knee: wx
        4: [0],      # right_knee: wx
        15: [0, 2],  # left_shoulder: wx, wz
        16: [0, 2],  # right_shoulder: wx, wz
        17: [1],     # left_elbow: wy
        18: [1],     # right_elbow: wy
    }
    
    selected_indices = [0, 1, 3, 4, 15, 16, 17, 18]  # 8개 관절
    
    # L2 norm 계산
    omega_l2_list = []
    for joint_idx in selected_indices:
        axes = joint_axis_config[joint_idx]
        # 해당 축들만 선택
        omega_joint = omega[:, :, joint_idx, axes]  # [bs, nframes-1, len(axes)]
        # L2 norm
        omega_l2 = torch.norm(omega_joint, dim=-1, keepdim=True)  # [bs, nframes-1, 1]
        omega_l2_list.append(omega_l2)
    
    # Stack: [bs, nframes-1, 8, 1]
    angular_velocity = torch.stack(omega_l2_list, dim=2)
    
    # 8. Shape 변환: [bs, nframes-1, 8, 1] → [bs, 8, 1, nframes-1]
    angular_velocity = angular_velocity.permute(0, 2, 3, 1)
    
    return angular_velocity