def compute_angular_velocity(self, motion_data, pose_rep='rot6d', mask=None, order='xyz'):
        """
        모션 데이터로부터 각속도(Angular Velocity), 각도 계산.
        노트북의 convert_6d_to_euler + np.diff 방식과 동일하게 구현.
        
        Args:
            motion_data: [bs, 263, 1, nframes] - 모션 데이터 (target 또는 model_output)
            pose_rep: str - Pose representation ('rot6d' 등)
            mask: [bs, 1, 1, nframes] - Padding mask (optional)
            order: str - Euler angle order ('xyz', 'zyx' 등)
        
        Returns:
            angular_velocity: [bs, 21, 3, nframes-1] - 각속도 (degree/frame)
        """
        # 1. 입력 데이터에서 rotation 부분 추출
        bs, _, _, nframes = motion_data.shape # motion_data: [bs, 263, 1, nframes]
        rot_data = motion_data[:, 67:193, 0, :].reshape(bs, nframes, 21, 6) # rotation data는 67:193 인덱스 (126차원 = 21 joints × 6D)
        
        # 2. 6D to Euler 변환
        rot_data_flat = rot_data.reshape(bs * nframes, 21, 6) # [bs*nframes, 21, 6]
        rot_matrix = cont6d_to_matrix(rot_data_flat) # [bs*nframes, 21, 3, 3]
        quat = matrix_to_quat(rot_matrix) # [bs*nframes, 21, 4]
        euler = qeuler(quat, order=order, deg=True).view(bs, nframes, 21, 3) # [bs, nframes, 21, 3]
        
        # 3. 차분 계산
        angular_velocity = euler[:, 1:, :, :] - euler[:, :-1, :, :] # [bs, nframes-1, 21, 3]

        # 5. Shape 변환
        return angular_velocity.permute(0, 2, 3, 1)  # [bs, 21, 3, nframes-1]