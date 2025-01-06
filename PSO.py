import random

class Particle:
    def __init__(self, bounds, initial_fitness, nv, mm,
                       w, c1, c2):
        self.particle_position = []
        self.particle_velocity = []
        self.local_best_particle_position = []
        self.fitness_local_best_particle_position = initial_fitness
        self.fitness_particle_position = initial_fitness
        self.nv = nv
        self.mm = mm
        ##### PARTICAL EQUATION #####
        self.w = w
        self.c1 = c1
        self.c2 = c2

        ## Score Equation ##
        self.total_trade = 0
        self.profit = 0
        self.max_absolute_drawdown = 0

        for i in range(self.nv):
            if type(bounds[i][0]) is int:
                self.particle_position.append(
                    random.randint(bounds[i][0], bounds[i][1])  # generate random intial position
                )
                self.particle_velocity.append(random.uniform(-1, 1))  # random initial velocity

            elif type(bounds[i][0]) is float:
                self.particle_position.append(
                    round(random.uniform(bounds[i][0], bounds[i][1]), 1)  # generate random intial position
                )
                self.particle_velocity.append(random.uniform(-1, 1))  # random initial velocity

    def evaluate(self, objective_function, 
                 start, end, l_interval, s_interval):
        # eat array and put the array to the objective function
        self.fitness_particle_position, self.total_trade, self.profit ,self.max_absolute_drawdown = objective_function(self.particle_position, start, end, l_interval, s_interval)
        if self.mm == -1:
            if self.fitness_particle_position < self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position  # update local best
                self.fitness_local_best_particle_position = self.fitness_particle_position  # update fitness of the local best
        if self.mm == 1:
            if self.fitness_particle_position > self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position  # position
                self.fitness_local_best_particle_position = self.fitness_particle_position  # score

    def update_velocity(self, global_best_particle_position):
        for i in range(self.nv):
            if type(self.particle_position[i]) is int:
                r1 = random.random()
                r2 = random.random()

                cognitive_velocity = self.c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
                social_velocity = self.c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
                self.particle_velocity[i] = int(self.w * self.particle_velocity[i] + cognitive_velocity + social_velocity)

            elif type(self.particle_position[i]) is float:
                r1 = random.random()
                r2 = random.random()

                cognitive_velocity = self.c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
                social_velocity = self.c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
                self.particle_velocity[i] = round(self.w * self.particle_velocity[i] + cognitive_velocity + social_velocity,
                                                  1)

    def update_position(self, bounds):
        for i in range(self.nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]
            # check match upper and lower bound
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][1]

            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][0]
