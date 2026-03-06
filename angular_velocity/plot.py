import matplotlib.pyplot as plt
import numpy as np
import os


def save_angular_velocity_plot(angular_velocity, out_path, rep_i, model_kwargs):
    """
    angular_velocity: [bs, 8, 1, nframes-1] - 8개 주요 관절의 각속도 L2 norm (rad/s)
    
    8개 관절:
    - left_hip, right_hip
    - left_knee, right_knee  
    - left_shoulder, right_shoulder
    - left_elbow, right_elbow
    """
    
    # 8개 선택된 관절 이름
    selected_joint_names = [
        "left_hip", "right_hip", 
        "left_knee", "right_knee",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow"
    ]
    
    # NumPy array로 변환 (tensor인 경우)
    if not isinstance(angular_velocity, np.ndarray):
        angular_velocity = angular_velocity.cpu().numpy()
    
    bs, n_joints, _, nframes_vel = angular_velocity.shape
    assert n_joints == 8, f"Expected 8 joints, got {n_joints}"
    
    # 각 동작별로 그래프 생성
    for b in range(bs):
        # 4행 2열 레이아웃 (좌/우 대칭)
        fig, axes = plt.subplots(4, 2, figsize=(14, 12))
        fig.suptitle(f'Angular Velocity L2 Norm (rad/s) - {model_kwargs["y"]["text"][b]}', 
                     fontsize=14, fontweight='bold')
        
        joint_pairs = [
            (0, 1),  # hip
            (2, 3),  # knee
            (4, 5),  # shoulder
            (6, 7),  # elbow
        ]
        
        for row, (left_idx, right_idx) in enumerate(joint_pairs):
            # 왼쪽 관절
            ax_left = axes[row, 0]
            omega_left = angular_velocity[b, left_idx, 0, :]  # [nframes-1]
            ax_left.plot(omega_left, color='blue', linewidth=1.5, label='‖ω‖')
            
            # 통계 계산
            avg_speed = np.mean(omega_left)
            max_speed = np.max(omega_left)
            
            ax_left.set_title(f'{selected_joint_names[left_idx]}', 
                             fontsize=11, fontweight='bold')
            ax_left.set_ylabel('rad/s', fontsize=9)
            ax_left.set_ylim(0.0, max(20.0, max_speed * 1.1))  # L2 norm은 항상 양수
            ax_left.legend(loc='upper right', fontsize=8)
            ax_left.grid(True, alpha=0.3, linestyle='--')
            
            # 통계 정보 표시
            info_left = f"Avg: {avg_speed:.2f} rad/s\nMax: {max_speed:.2f} rad/s"
            ax_left.text(0.02, 0.98, info_left,
                        transform=ax_left.transAxes,
                        ha='left', va='top',
                        fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.75, 
                                edgecolor='none', boxstyle='round,pad=0.3'))
            
            # 오른쪽 관절
            ax_right = axes[row, 1]
            omega_right = angular_velocity[b, right_idx, 0, :]  # [nframes-1]
            ax_right.plot(omega_right, color='red', linewidth=1.5, label='‖ω‖')
            
            # 통계 계산
            avg_speed = np.mean(omega_right)
            max_speed = np.max(omega_right)
            
            ax_right.set_title(f'{selected_joint_names[right_idx]}', 
                              fontsize=11, fontweight='bold')
            ax_right.set_ylabel('rad/s', fontsize=9)
            ax_right.set_ylim(0.0, max(20.0, max_speed * 1.1))
            ax_right.legend(loc='upper right', fontsize=8)
            ax_right.grid(True, alpha=0.3, linestyle='--')
            
            # 통계 정보 표시
            info_right = f"Avg: {avg_speed:.2f} rad/s\nMax: {max_speed:.2f} rad/s"
            ax_right.text(0.02, 0.98, info_right,
                         transform=ax_right.transAxes,
                         ha='left', va='top',
                         fontsize=10, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.75, 
                                 edgecolor='none', boxstyle='round,pad=0.3'))
            
            # 하단 행에만 x축 레이블
            if row == 3:
                ax_left.set_xlabel('Frame', fontsize=9)
                ax_right.set_xlabel('Frame', fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(out_path, f'angular_velocity_l2norm_rep{rep_i:02d}_motion{b}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'  Saved Angular velocity L2 norm plot for motion {b}')
    

def save_euler_angles_plot(euler_angles, out_path, rep_i, model_kwargs):
    """
    euler_angles: [bs, 21, 3, nframes] - 각 관절의 오일러 각도 (order에 따라 다름)
    """

    # 관절 이름 정의
    joint_names = [
        "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
        "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist"
    ]
    
    # 이미 numpy array인 경우 그대로, tensor인 경우 변환
    if not isinstance(euler_angles, np.ndarray):
        euler_angles = euler_angles.cpu().numpy()
    
    bs, n_joints, n_axes, nframes = euler_angles.shape
    
    # 각 동작별로 그래프 생성
    for b in range(bs):
        fig, axes = plt.subplots(7, 3, figsize=(18, 24))  # 21 joints = 7 rows × 3 cols
        #fig.suptitle(f'Euler Angles - {model_kwargs["y"]["text"][b]}', fontsize=16)
        
        for joint_idx in range(n_joints):
            row = joint_idx // 3
            col = joint_idx % 3
            ax = axes[row, col]
            
            # 3개 축의 각도 플롯 (axis 0, 1, 2)
            ax.plot(euler_angles[b, joint_idx, 2, :], label='Z', alpha=0.7, linewidth=1.5)
            ax.plot(euler_angles[b, joint_idx, 0, :], label='X', alpha=0.7, linewidth=1.5)
            ax.plot(euler_angles[b, joint_idx, 1, :], label='Y', alpha=0.7, linewidth=1.5)
            
            # 각속도 계산 (프레임 간 차분)
            angular_vel = np.diff(euler_angles[b, joint_idx, :, :], axis=1)  # [3, nframes-1]
            
            # 각 축의 각속도 크기를 L2 norm으로 계산
            angular_speed = np.linalg.norm(angular_vel, axis=0)  # [nframes-1]
            avg_speed = np.mean(angular_speed)  # 스칼라 평균값
            
            avg_speed_x = np.mean(np.abs(angular_vel[0, :]))  # X축 평균 각속도
            avg_speed_y = np.mean(np.abs(angular_vel[1, :]))  # Y축 평균 각속도
            avg_speed_z = np.mean(np.abs(angular_vel[2, :]))  # Z축 평균 각속도
                        
            ax.set_title(f'{joint_names[joint_idx]} (Joint {joint_idx})', fontsize=10, fontweight='bold')

            # 플롯 안에 크게 표시할 텍스트
            info = (
                f"Avg: {avg_speed:.2f}°/f\n"
                f"X: {avg_speed_x:.2f}  Y: {avg_speed_y:.2f}  Z: {avg_speed_z:.2f}°/f"
            )

            # 플롯 내부(좌상단)에 크게 쓰기
            ax.text(
                0.02, 0.98, info,                 # (x,y) in axes fraction
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=13, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.3')
            )
            
            ax.set_xlabel('Frame', fontsize=9)
            ax.set_ylabel('Angle (deg)', fontsize=9)
            ax.set_ylim(-180, 180)  # Euler angles typically range from -180 to 180 degrees
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.tick_params(labelsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # 타이틀 공간 확보
        save_path = os.path.join(out_path, f'euler_angles_rep{rep_i:02d}_motion{b}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'  Saved Euler angles plot for motion {b}')
    
def save_euler_velocity_plot(angular_velocity, out_path, rep_i, model_kwargs):
    """
    angular_velocity: [bs, 21, 3, nframes-1] - 각 관절의 각속도 (degree/frame)
    """

    # 관절 이름 정의
    joint_names = [
        "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
        "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist"
    ]
    
    # 이미 numpy array인 경우 그대로, tensor인 경우 변환
    if not isinstance(angular_velocity, np.ndarray):
        angular_velocity = angular_velocity.cpu().numpy()
    
    bs, n_joints, n_axes, nframes_vel = angular_velocity.shape
    
    # 각 동작별로 그래프 생성
    for b in range(bs):
        fig, axes = plt.subplots(7, 3, figsize=(18, 24))  # 21 joints = 7 rows × 3 cols
        fig.suptitle(f'Angular Velocity (deg/frame) - {model_kwargs["y"]["text"][b]}', fontsize=16)
        
        for joint_idx in range(n_joints):
            row = joint_idx // 3
            col = joint_idx % 3
            ax = axes[row, col]
            
            # 3개 축의 각속도 플롯 (axis 0, 1, 2)
            ax.plot(angular_velocity[b, joint_idx, 2, :], label='Z', alpha=0.7, linewidth=1.5)
            ax.plot(angular_velocity[b, joint_idx, 0, :], label='X', alpha=0.7, linewidth=1.5)
            ax.plot(angular_velocity[b, joint_idx, 1, :], label='Y', alpha=0.7, linewidth=1.5)
            
            # 관절 이름으로 제목 설정
            ax.set_title(f'{joint_names[joint_idx]} (Joint {joint_idx})', fontsize=10, fontweight='bold')
            ax.set_xlabel('Frame', fontsize=9)
            ax.set_ylabel('Angular Velocity (deg/frame)', fontsize=9)
            ax.set_ylim(-90, 90)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.tick_params(labelsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # 타이틀 공간 확보
        save_path = os.path.join(out_path, f'angular_velocity_rep{rep_i:02d}_motion{b}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'  Saved Angular velocity plot for motion {b}')
    
def save_quaternion_plot(quaternions, out_path, rep_i, model_kwargs):
    """
    quaternions: [bs, nframes, 21, 4] - 각 관절의 quaternion (w, x, y, z)
    """
    bs, nframes, n_joints, _ = quaternions.shape
    
    joint_names = [
        "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
        "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist"
    ]
    
    for b in range(bs):
        # 21개 관절을 7x3 grid로 배치
        fig, axes = plt.subplots(7, 3, figsize=(18, 24))
        fig.suptitle(f'Quaternions - {model_kwargs["y"]["text"][b]}', fontsize=16)
        
        for joint_idx in range(n_joints):
            row = joint_idx // 3
            col = joint_idx % 3
            ax = axes[row, col]
            
            # w, x, y, z 성분 플롯
            ax.plot(quaternions[b, :, joint_idx, 0], label='w', alpha=0.7, linewidth=1.5)
            ax.plot(quaternions[b, :, joint_idx, 1], label='x', alpha=0.7, linewidth=1.5)
            ax.plot(quaternions[b, :, joint_idx, 2], label='y', alpha=0.7, linewidth=1.5)
            ax.plot(quaternions[b, :, joint_idx, 3], label='z', alpha=0.7, linewidth=1.5)
            
            ax.set_title(f'{joint_names[joint_idx]} (Joint {joint_idx})', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Frame', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.set_ylim(-1, 1)  # Quaternion은 -1 ~ 1 범위
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.tick_params(labelsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = os.path.join(out_path, f'quaternion_rep{rep_i:02d}_motion{b}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'  Saved quaternion plot for motion {b}')
        
def save_rotation_matrix_plot(rotation_matrices, out_path, rep_i, model_kwargs):
    """
    rotation_matrices: [bs, nframes, 21, 3, 3] - 각 관절의 rotation matrix
    """
    bs, nframes, n_joints, _, _ = rotation_matrices.shape
    
    joint_names = [
        "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
        "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist"
    ]
    
    for b in range(bs):
        for joint_idx in range(n_joints):
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            fig.suptitle(f'{joint_names[joint_idx]} (Joint {joint_idx}) - Rotation Matrix\n{model_kwargs["y"]["text"][b]}', 
                         fontsize=14, fontweight='bold')
            
            for row in range(3):
                for col in range(3):
                    ax = axes[row, col]
                    element_values = rotation_matrices[b, :, joint_idx, row, col]  # [nframes]
                    
                    ax.plot(element_values, linewidth=1.5, color='steelblue')
                    ax.set_title(f'R[{row},{col}]', fontsize=11, fontweight='bold')
                    ax.set_xlabel('Frame', fontsize=9)
                    ax.set_ylabel('Value', fontsize=9)
                    ax.set_ylim(-1, 1)  # y축 범위를 -1 ~ 1로 고정
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
                    ax.tick_params(labelsize=8)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = os.path.join(out_path, f'rotmat_rep{rep_i:02d}_motion{b}_joint{joint_idx:02d}_{joint_names[joint_idx]}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        print(f'  Saved rotation matrix plots for motion {b}')