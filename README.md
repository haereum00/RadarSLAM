# Radar Odometry, SLAM and 3D Occupancy with IWR6843ISK

[Radarize (MobiSys'24)](https://github.com/ConnectedSystemsLab/radarize_ae)를 TI IWR6843ISK 레이더와 Livox Mid-360 라이다 장비로 옮기고, 레이더 3D 점유 예측을 추가한 코드입니다.

## 파이프라인

```
녹화 (.bag): 레이더 raw ADC + Mid-360 + IMU + FAST-LIO2
  │
  ├─ tools/create_dataset.py      → data/seqXX.npz     레이더 히트맵, 2D 정답 (속도, 회전, 벽)
  └─ tools/create_dataset_3d.py   → data/seqXX_3d.npz  레이더 높이 단서, 3D 정답 (높이 6개 층)
  │
학습 (PyTorch)
  ├─ tools/train_flow.py     ResNet-18       레이더 Doppler → 속도
  ├─ tools/train_rot.py      ECA-ResNet-18   레이더 히트맵 → 회전
  ├─ tools/train_unet.py     U-Net           레이더 히트맵 → 2D 벽
  └─ tools/train_unet3d.py   U-Net (6층)     레이더 히트맵 + 높이 단서 → 3D 장애물
  │
테스트:   tools/test_*.py, tools/test_odom.py (속도 + 회전 → 궤적)
SLAM:     cartographer/ros2/run_carto_ros2.sh (Cartographer, ROS 2 Humble)
3D 지도:  tools/map3d.py → .ply
```

정답은 학습할 때만 쓰고, 테스트할 때 모델 입력은 레이더뿐입니다.
- 위치 정답: FAST-LIO2 궤적
- 벽, 장애물 정답: Mid-360 점을 FAST-LIO2 위치로 쌓은 지도

## 원본 Radarize에서 바꾼 점

- **정답 생성**: RealSense T265/D435 대신 FAST-LIO2 궤적과 Mid-360 지도. Mid-360 IMU의 중력 방향으로 수평을 맞추고, 레이더 시야를 모두 채우도록 앞뒤 15초 스캔을 누적합니다 (`tools/create_dataset.py`).
- **레이더**: IWR6843ISK는 송신 순서가 1843과 달라서 가상 안테나 채널 순서를 바꿉니다 (`RADAR_CHANNEL_ORDER`).
- **SLAM**: 원본은 ROS 1 Cartographer를 썼습니다. Ubuntu 24.04 서버에서 돌리기 위해 ROS 2 Humble 버전으로 바꿨습니다 (`cartographer/ros2/`).

## 3D 확장

- **입력**: 기존 6채널(거리-각도 히트맵)에 레이더 높이 단서 2채널을 더합니다. IWR6843ISK는 TX1 안테나가 반 파장 위에 있어서, 같은 방위로 맞춘 위·아래 안테나 줄의 위상차가 반사점의 높이 각도가 됩니다. TDM 송신 시차로 생기는 Doppler 위상은 먼저 제거합니다 (`create_radar_elevation`).
- **정답**: 라이다 지도를 레이더 높이 기준 −1.3 ~ +1.0 m, 0.4 m 간격 6개 층으로 자릅니다. 바닥(−1.4 m)과 천장(+1.1 m)은 항상 차 있어서 제외했습니다 (`create_lidar_3d`).
- **모델**: 2D와 같은 U-Net 구조이고, 출력이 층마다 독립 확률(sigmoid) 6장입니다. 손실은 BCE + Dice입니다 (`UNet3D`).

## 결과

시퀀스 11개(한 층 T자 복도, 총 27분). 학습 8개, 검증 seq17, 테스트 seq13 / seq20.

| 항목 | seq13 | seq20 |
|---|---|---|
| 속도 RMSE | 0.048 m/s (두 시퀀스 합) | |
| 회전 RMSE (4프레임) | 0.033 rad | 0.048 rad |
| 2D 벽 거리 오차 (중앙값) | 14 cm | 9 cm |
| 위치 오차 ATE, 레이더 오도메트리 | 6.46 m | 2.84 m |
| 위치 오차 ATE, 레이더 SLAM | 9.45 m | **1.25 m** |
| **3D 장애물 F1** | **0.82** | **0.73** |
| 3D 비교: 2D 예측을 높이마다 복사 | 0.60 | 0.48 |
| 3D 비교: 라이다 2D 정답을 높이마다 복사 | 0.66 | 0.59 |

F1은 이웃 1칸(약 5 cm, 1°) 허용 기준입니다. 3D 모델이 라이다 2D 정답을 복사한 것보다도 높아서, 레이더에서 높이별 구조를 읽어 낸다는 것을 보여 줍니다.

한계
- 데이터가 같은 경로 11개뿐이라 회전 추정이 약하고, 긴 경로(seq13)에서는 SLAM이 실패합니다.
- 위치 추정과 SLAM은 2D이고, 3D는 장애물 예측에만 해당합니다.

## 실행

```bash
./run_rig2.sh          # 변환 → 학습 / 테스트 → SLAM → 3D 지도
./run_rig2.sh train    # 변환 생략
```
- 데이터: `data/seqXX.bag`. 분할은 `configs/rig2.yaml`
- 결과: `output_rig2/`
- 장비 설정: `calib/6843isk/` (레이더 cfg), `calib/mid360_upright/` (라이다 → 레이더 extrinsic)

### 환경

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

Cartographer (ROS 2 Humble, RoboStack, root 권한 불필요):
```bash
micromamba create -n ros_carto -c conda-forge -c robostack-staging \
    ros-humble-cartographer-ros ros-humble-ros2bag ros-humble-rosbag2-storage-default-plugins \
    ros-humble-ros-core python=3.11
micromamba install -n ros_carto -c conda-forge "protobuf=5.28" rosbags
```

## 녹화 조건

- Mid-360은 돔이 위로 오게 장착 (앞으로 눕히면 정면 사각지대 때문에 FAST-LIO2가 실패)
- 녹화마다 FAST-LIO2 재시작, 시작 후 2초 정지
- 토픽: `/radar0/radar_data`, `/livox/lidar`, `/livox/imu`, `/Odometry`
