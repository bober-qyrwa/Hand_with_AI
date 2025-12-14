import torch.nn.functional as F
from pyTorch_model import *
from main import *
frame =0
sensor_data = 0



class RewardController(nn.Module):
    def __init__(self):
        super().__init__()
        # Carrot & Stick RL controller [web:13][web:18]
        self.policy_net = nn.Sequential(
            nn.Linear(63, 128),  # 21*3 (3D pose error)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)  # 12 сервоприводов стенда
        )

    def forward(self, pose_error_3d):
        # Reward: -||target - current||^2 (carrot), penalty за большие движения (stick)
        actions = torch.tanh(self.policy_net(pose_error_3d))  # Углы [-1,1]

        # Reward shaping
        reward = -torch.norm(pose_error_3d) + 0.1 * torch.norm(actions)
        return actions, reward


# Инференс полного пайплайна
controller = RewardController()
# Полный цикл
hand_kpts = model_hand.predict(frame)[0].keypoints.xy  # 2D hand [web:1]
body_kpts = model_coco.predict(frame)[0].keypoints.xy  # 2D body [web:3]
pose_3d = model_fusion(hand_kpts, sensor_data)  # 3D fusion [web:16]
target_3d = torch.tensor([...])  # Целевая поза
error = target_3d - pose_3d
servo_angles, reward = controller(error)  # Команды приводам [web:17]

print(f"Servo angles: {servo_angles}, Reward: {reward}")
