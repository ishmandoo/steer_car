import numpy as np
import pygame
import time


class SteerCar:
    def __init__(self):
        self.state = None
        self.dt = 0.1

        self.speed = 1. # meters per second
        self.turn_radius = 20. # meters
        self.length = 10. # meters

        self.screen = None

    def reset(self):
        self.state = np.array([0.,0.,0.])

    def step(self, action):
        # accel = -1, 0, or 1
        # steer = -1, 0, or 1
        accel, steer = action

        x, y, theta = self.state
        pos = np.array([x,y])

        dtheta = steer * accel * (self.speed / self.turn_radius) * self.dt
        dx, dy = accel * self.speed * np.array([np.cos(theta), np.sin(theta)]) * self.dt

        self.state += np.array([dx, dy, dtheta])

    def render(self):
        if self.screen is None:
            size = [400, 300]
            self.screen = pygame.display.set_mode(size)

        w, h = pygame.display.get_surface().get_size()

        self.screen.fill((0,0,0))

        x, y, theta = self.state

        back_pos = np.array([x,y])
        back_pos += np.array([w/2, h/2])
        front_pos = back_pos + self.length * np.array([np.cos(theta), np.sin(theta)])

        pygame.draw.line(self.screen, (0,255,0), back_pos, front_pos, 5)

        pygame.display.flip()
        


'''
car = SteerCar()
car.reset()
target = 2 * np.pi-1.
for i in range(2000):
    if np.abs(car.state[2] - target) > 0.1:
        car.step([1.,1.])
    else:
        car.step([0.,0.])

    car.render()
    time.sleep(0.01)
'''
