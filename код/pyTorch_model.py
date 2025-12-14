import torch
import torch.nn as nn
from torch.optim import Adam
yolo_2d = 0
sensor_angles = 0
real_3d_sensor = 0



class KinematicsFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 2D keypoints → 3D projection + sensor fusion
        self.fc1 = nn.Linear(42, 128)  # 21 hand keypoints * 2D (x,y)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 21 * 3)  # 21 keypoints * 3D (x,y,z)

    def forward(self, keypoints_2d, sensor_data):
        # Forward kinematics projection [web:11][web:14]
        x = torch.relu(self.fc1(torch.cat([keypoints_2d, sensor_data], dim=1)))
        x = torch.relu(self.fc2(x))
        pose_3d = self.fc3(x)  # Предсказанная 3D поза

        # MSE loss между predicted и real sensor positions
        return pose_3d


# Обучение
model_fusion = KinematicsFusion()
optimizer = Adam(model_fusion.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(100):
    pred_3d = model_fusion(yolo_2d, sensor_angles)
    loss = criterion(pred_3d, real_3d_sensor)
    loss.backward()
    optimizer.step()
