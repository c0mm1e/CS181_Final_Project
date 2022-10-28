from ast import While
import sc2
from sc2 import run_game, maps, Race, Difficulty, Result
from sc2.player import Bot, Computer
from sc2 import position
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY, ROBOTICSFACILITY, OBSERVER, \
 ZEALOT, STALKER, COLOSSUS, MOTHERSHIP, FORGE, FLEETBEACON, TWILIGHTCOUNCIL, \
 PHOTONCANNON, TEMPLARARCHIVE, DARKSHRINE, ROBOTICSBAY, HIGHTEMPLAR, DARKTEMPLAR, \
 SENTRY, PHOENIX, CARRIER, WARPPRISM, IMMORTAL, INTERCEPTOR, WARPGATE, \
 FORCEFIELD, WARPPRISMPHASING, ARCHON, ADEPT, ORACLE, TEMPEST, DISRUPTOR, \
 ORACLESTASISTRAP, DISRUPTORPHASED, ADEPTPHASESHIFT, SHIELDBATTERY, \
 OBSERVERSIEGEMODE, ASSIMILATORRICH
import random
import cv2
import numpy as np
import os
import time
import math
import keras
from scipy.fftpack import diff

# Bot class inherits from sc2.BotAI
class ChungBot(sc2.BotAI):
    def __init__(self, bot_race, bot_difficulty, use_model=False, model_name=None, visualization_window=True):
        self.bot_race = bot_race
        self.bot_difficulty = bot_difficulty
        self.visualization_window = visualization_window
        self.use_model = use_model
        self.model_name = model_name
        self.scouts_and_spots = {}
        # All choices AI can choose
        self.choices = {0: self.protect_base,
                        1: self.attack_unit,
                        2: self.attack_structure, 
                        3: self.attack_start_point, 
                        }
        # To save training data
        self.train_data = []
        # When use_model=True, use the model to predict the next action
        if self.use_model:
            print("USING MODEL")
            self.model = keras.models.load_model(model_name)

    # This function will be called when game is over
    def on_end(self, game_result):
        print(game_result, self.use_model)
        # Only save winning game data
        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))
        # Record wins and losses
        if self.bot_race and self.bot_difficulty:
            with open("{}_{}.txt".format(self.bot_race, self.bot_difficulty),"a") as f:
                if self.use_model:
                    f.write("Model {}\n".format(game_result))
                else:
                    f.write("Random {}\n".format(game_result))

    # This function will be called once in each game_loop (about 45ms)
    async def on_step(self, iteration):
        # The number of minutes game has been played
        # Use self.time to get the current time (second)
        print('Time:',self.time/60)
        # Call functions
        try:
            await self.distribute_workers() # built-in function
        except Exception as e:
            print(str(e))
        await self.scout()
        await self.visualization()
        await self.make_choice()
        await self.build_assimilators()
        await self.build_barracks()
        await self.build_pylons()
        await self.expand()
        await self.train_workers()
        await self.train_voidrays()
        await self.train_observers()

    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))
    
    async def build_barracks(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon.position.towards(self.game_info.map_center, 5))
            elif not self.units(CYBERNETICSCORE):
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon.position.towards(self.game_info.map_center, 5))
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(ROBOTICSFACILITY)) < 1:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon.position.towards(self.game_info.map_center, 5))
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < self.time / 60:
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_pylons(self):
        if self.supply_left < 6 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))
    
    async def expand(self):
        if self.units(NEXUS).amount < self.time / 120 and self.can_afford(NEXUS):
            await self.expand_now()

    # Train workers
    async def train_workers(self):
        if (len(self.units(NEXUS)) * 16) > len(self.units(PROBE)) and len(self.units(PROBE)) < 65:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))
    # Train voidrays
    async def train_voidrays(self):
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 1:
                await self.do(sg.train(VOIDRAY))
    # Train observers
    async def train_observers(self):
        if len(self.units(OBSERVER)) < self.time / 180:
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))

    def random_location_variance(self, location):
        x = location[0] + random.randrange(-5,5)
        y = location[1] + random.randrange(-5,5)
        # Make sure the coordinate is valid
        if x < 0: x = 0
        if y < 0:  y = 0
        if x > self.game_info.map_size[0]: x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]: y = self.game_info.map_size[1]
        return position.Point2(position.Pointlike((x,y)))

    # Scout the enemy base
    async def scout(self):
        # Store the distance between enemy's start location and all possible sub-base locations
        self.expand_dis_dir = {}
        # self.expansion_locations is all possible sub-base locations
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            self.expand_dis_dir[distance_to_enemy_start] = el
        # Sorted by distance
        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_units = [unit.tag for unit in self.units]
        # Update scouts list
        # Removing scouts that are actually dead now.
        scouts_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_units:
                scouts_removed.append(noted_scout)
        for scout in scouts_removed:
            del self.scouts_and_spots[scout]
        # When cannot train observer, send a probe to scout
        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type, unit_limit = PROBE, 1
        # Else send observers to scout
        else:
            unit_type, unit_limit = OBSERVER, 10
        # Send scouts to all possible base location
        assign_scout = True
        if unit_type == PROBE:
            for probe in self.units(PROBE):
                if probe.tag in self.scouts_and_spots:
                    assign_scout = False
        if assign_scout:
            if len(self.units(unit_type).idle):
                for scout in self.units(unit_type).idle[:unit_limit]:
                    if scout.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = next(value for key, value in self.expand_dis_dir.items() if key == dist)
                                active_locations = [self.scouts_and_spots[i] for i in self.scouts_and_spots]
                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue
                                    await self.do(scout.move(location))
                                    self.scouts_and_spots[scout.tag] = location
                                    break
                            except:
                                pass
        # If scout is probe, randomly move
        for scout in self.units(unit_type):
            if scout.tag in self.scouts_and_spots:
                if scout in self.units(PROBE):
                    await self.do(scout.move(self.random_location_variance(self.scouts_and_spots[scout.tag])))

    # Visualize the map
    async def visualization(self):
        # The size of map AutomatonLE is 148x148
        # The size of map AutomatonLE in game_info.map_size is 200x176
        # game_data is a color image (BGR)
        game_data = np.zeros((self.game_info.map_size[1] + 20, self.game_info.map_size[0], 3), np.uint8)
        print(game_data.shape)
        # Own units
        draw_dict = {
                    # Base (Nexus): Green
                    NEXUS: [15, (0, 255, 0)],
                    # Barrack: Blue
                    GATEWAY: [3, (200, 100, 0)],
                    STARGATE: [3, (200, 100, 0)],
                    # Combat unit: Dark green
                    VOIDRAY: [1, (255, 100, 0)],
                    # Observer: White
                    OBSERVER: [1, (255, 255, 255)],
                    }
        # Draw circles on the map
        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                cv2.circle(game_data, (int(unit.position[0]), int(unit.position[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)
        # Draw the enemy bases
        base_names = ["nexus", "commandcenter", "orbitalcommand", "planetaryfortress", "hatchery", "lair", "hive"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)
        # Draw the enemy structures except bases
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (200, 50, 212), -1)
        # Draw the enemy combat units
        for enemy_unit in self.known_enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ["probe", "scv", "drone"]
                pos = enemy_unit.position
                if enemy_unit.name.lower() not in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (50, 0, 215), -1)
        # line_max is equal to the width of the map
        line_max = self.game_info.map_size[0]
        # Draw the number of minerals
        mineral_ratio = min(self.minerals / 1500, 1.0)
        cv2.line(game_data, (0, self.game_info.map_size[1] + 17), (int(line_max*mineral_ratio), self.game_info.map_size[1] + 17), (0, 255, 25), 3)
        # Draw the number of vespene gas
        vespene_ratio = min(self.vespene / 1500, 1.0)
        cv2.line(game_data, (0, self.game_info.map_size[1] + 13), (int(line_max*vespene_ratio), self.game_info.map_size[1] + 13), (210, 200, 0), 3)
        # Draw the number of used supply
        population = self.supply_used / 200.0
        cv2.line(game_data, (0, self.game_info.map_size[1] + 9), (int(line_max*population), self.game_info.map_size[1] + 9), (150, 150, 150), 3)
        # Draw the number of supply limit
        supply = self.supply_cap / 200.0
        cv2.line(game_data, (0, self.game_info.map_size[1] + 5), (int(line_max*supply), self.game_info.map_size[1] + 5), (220, 200, 200), 3)
        # Draw the ratio of worker to supply
        worker_ratio = len(self.units(PROBE)) / self.supply_used
        cv2.line(game_data, (0, self.game_info.map_size[1] + 1), (int(line_max*worker_ratio), self.game_info.map_size[1] + 1), (250, 250, 200), 3)

        # flip horizontally to make our final fix in visual representation:
        self.flipped = cv2.flip(game_data, 0)
        # Make a window to display
        if self.visualization_window:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Visualization', resized)
            cv2.waitKey(1)

    # There are all choices
    # Choice 0: Attack enemy unit which is closest to nexus (protect nexus)
    async def protect_base(self):
        if len(self.known_enemy_units):
            target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
            min_distance = min([target.distance_to(self.units(NEXUS)[i]) for i in range(len(self.units(NEXUS)))])
            if min_distance < 15:
                for u in self.units(VOIDRAY).idle:
                    await self.do(u.attack(target))
            else:
                self.valid_choice = False
        else:
            self.valid_choice = False

    # Choice 1: Attack a known enemy unit
    async def attack_unit(self):
        if len(self.known_enemy_units):
            target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
            for u in self.units(VOIDRAY).idle:
                await self.do(u.attack(target))
        else:
            self.valid_choice = False

    # Choice 2: Attack a known enemy structure
    async def attack_structure(self):
        if len(self.known_enemy_structures):
            target = self.known_enemy_structures.closest_to(random.choice(self.units(NEXUS)))
            for u in self.units(VOIDRAY).idle:
                await self.do(u.attack(target))
        else:
            self.valid_choice = False

    # Choice 3: Attack start point
    async def attack_start_point(self):
        target = self.enemy_start_locations[0]
        for u in self.units(VOIDRAY).idle:
            await self.do(u.attack(target))

    # Make choice
    async def make_choice(self):
        if len(self.units(VOIDRAY).idle):
            self.valid_choice = True
            if self.use_model:
                prediction = self.model.predict([self.flipped.reshape([-1, 196, 200, 3])])
                choice = np.argmax(prediction[0])
            else:
                choice = random.randint(0, len(self.choices)-1)
            # Do thing which is chosen
            await self.choices[choice]()
            # Label
            label = np.zeros(len(self.choices))
            label[choice] = 1
            # A data pair
            if self.valid_choice:
                self.train_data.append([label, self.flipped])

# Continuously generate training data
# while True:
#     race = random.choice([Race.Protoss, Race.Terran, Race.Zerg])
#     difficulty = Difficulty.Medium
#     # race = None
#     # difficulty = None
#     run_game(maps.get("AbyssalReefLE"), [
#         Bot(Race.Protoss, ChungBot(race, difficulty, use_model=False, model_name=None, visualization_window=False)),
#         # Bot(Race.Protoss, ChungBot(race, difficulty, use_model=False, model_name=None, visualization_window=False))
#         Computer(race, difficulty)
#         ], realtime=False)

for i in range(17):
    print("----------Now it is the " + str(i+1) + "th Easy game----------")
    race = random.choice([Race.Protoss, Race.Terran, Race.Zerg])
    difficulty = Difficulty.Easy
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, ChungBot(race, difficulty, use_model=True, model_name="epochs-0", visualization_window=False)),
        # Bot(Race.Protoss, ChungBot(race, difficulty, use_model=False, model_name=None, visualization_window=False))
        Computer(race, difficulty)
        ], realtime=False)

for i in range(17):
    print("----------Now it is the " + str(i+1) + "th Hard game----------")
    race = random.choice([Race.Protoss, Race.Terran, Race.Zerg])
    difficulty = Difficulty.Hard
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, ChungBot(race, difficulty, use_model=True, model_name="epochs-0", visualization_window=False)),
        # Bot(Race.Protoss, ChungBot(race, difficulty, use_model=False, model_name=None, visualization_window=False))
        Computer(race, difficulty)
        ], realtime=False)

# Only play once, used for testing
# run_game(maps.get("AbyssalReefLE"), [
#     Bot(Race.Protoss, ChungBot(Race.Protoss, Difficulty.Medium, use_model=False, model_name=None, visualization_window=True)),
#     Computer(Race.Protoss, Difficulty.Medium)
#     ], realtime=False)