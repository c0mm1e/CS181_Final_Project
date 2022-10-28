import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY
import random
import cv2
import numpy as np
import time
import os

data = np.load(os.path.join('train_data', 't1530807965.npy'), allow_pickle=True)
print(data.shape)
print(data)
print('--------------------------------')
print(data[0])