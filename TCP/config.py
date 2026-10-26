import os

class GlobalConfig:
	""" base architecture configurations """
	# Data
	seq_len = 1 # input timesteps
	pred_len = 4 # future waypoints predicted

	# data root
	root_dir_all = "data"

	# train_towns = ['town01', 'town03', 'town04',  'town06', ]
	# val_towns = ['town02', 'town05', 'town07', 'town10']
	train_towns = ['town01']
	val_towns = ['town01']
	train_data, val_data = [], []
	for town in train_towns:
		train_data.append(os.path.join(root_dir_all, town))
		# train_data.append(os.path.join(root_dir_all, town+'_addition'))
	for town in val_towns:
		# val_data.append(os.path.join(root_dir_all, town+'_val'))
		val_data.append(os.path.join(root_dir_all, town))

	ignore_sides = True # don't consider side cameras
	ignore_rear = True # don't consider rear cameras

	input_resolution = 256

	scale = 1 # image pre-processing
	crop = 256 # image pre-processing

	# ---- v3 标定参数（来源：leaderboard/team_code/tcp_agent.py sensors()）----
	# 相机：type=sensor.camera.rgb，900x256，fov=100，位姿 x=-1.5 z=2.0，roll=pitch=yaw=0
	img_w = 900
	img_h = 256
	cam_fov = 100.0
	cam_pos = (-1.5, 0.0, 2.0)
	# LiDAR：type=sensor.lidar.ray_cast，位姿 x=1.3 z=2.5，yaw=-90
	lidar_pos = (1.3, 0.0, 2.5)
	lidar_yaw = -90.0

	lr = 1e-4 # learning rate

	# Controller
	turn_KP = 0.75
	turn_KI = 0.75
	turn_KD = 0.3
	turn_n = 40 # buffer size

	speed_KP = 5.0
	speed_KI = 0.5
	speed_KD = 1.0
	speed_n = 40 # buffer size

	max_throttle = 0.75 # upper limit on throttle signal value in dataset
	brake_speed = 0.4 # desired speed below which brake is triggered
	brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
	clip_delta = 0.25 # maximum change in speed input to logitudinal controller


	aim_dist = 4.0 # distance to search around for aim point
	angle_thresh = 0.3 # outlier control detection angle
	dist_thresh = 10 # target point y-distance for outlier filtering


	speed_weight = 0.05
	value_weight = 0.001
	features_weight = 0.05

	rl_ckpt = "roach/log/ckpt_11833344.pth"

	img_aug = True


	def __init__(self, **kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)
