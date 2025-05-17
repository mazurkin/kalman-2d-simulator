import pygame
import numpy as np
import collections


class KalmanFilter:

    def __init__(self, dt: float, process_noise_std: float, measurement_noise_std: float):
        self.dt = dt
        self.x = np.zeros((6, 1))

        dt2 = dt ** 2 / 2
        self.A = np.array([
            [1,   0,  dt,   0, dt2,   0],
            [0,   1,   0,  dt,   0, dt2],
            [0,   0,   1,   0,  dt,   0],
            [0,   0,   0,   1,   0,  dt],
            [0,   0,   0,   0,   1,   0],
            [0,   0,   0,   0,   0,   1],
        ])

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        self.Q = process_noise_std ** 2 * np.eye(6)
        self.R = measurement_noise_std ** 2 * np.eye(2)
        self.P = np.eye(6)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2]

    def update(self, z):
        z = np.reshape(z, (2, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x

class Simulator:

    # dimension of the window
    WIDTH, HEIGHT = 1024, 768

    # time step in virtual units
    DT: float = 0.1

    # display scale
    SCALE: float = 1.0

    # increment of the acceleration
    ACC_INCREMENT: float = 0.02

    # limit of the acceleration
    ACC_LIMIT: float = 1.0

    # fps limit
    FPS: int = 30

    # measurement
    MEASUREMENTS: int = FPS

    # gaussian noise of process
    PROCESS_NOISE_STD: float = 1.0 * ACC_INCREMENT

    # gaussian noise of measurements
    MEASUREMENT_NOISE_STD: float = 10.0

    def __init__(self):
        self.pos: np.array = np.array([0.0, 0.0], dtype=np.float64)
        self.vel: np.array = np.array([0.0, 0.0], dtype=np.float64)
        self.acc: np.array = np.array([0.0, 0.0], dtype=np.float64)

        self.kf = KalmanFilter(self.DT, self.PROCESS_NOISE_STD, self.MEASUREMENT_NOISE_STD)

        self.last_x = collections.deque(maxlen=self.MEASUREMENTS)
        self.last_x.extend(np.zeros(self.MEASUREMENTS))
        self.last_y = collections.deque(maxlen=self.MEASUREMENTS)
        self.last_y.extend(np.zeros(self.MEASUREMENTS))

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Kalman Filter 2D Simulation")

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Courier', 20)

    def loop(self):
        running = True

        while running:
            self.screen.fill((255, 255, 255))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_RIGHT]:
                if self.acc[0] >= 0.0:
                    self.acc[0] += self.ACC_INCREMENT
                else:
                    self.acc[0] = 0.0
            if keys[pygame.K_LEFT]:
                if self.acc[0] <= 0.0:
                    self.acc[0] -= self.ACC_INCREMENT
                else:
                    self.acc[0] = 0.0
            if keys[pygame.K_UP]:
                if self.acc[1] >= 0.0:
                    self.acc[1] += self.ACC_INCREMENT
                else:
                    self.acc[1] = 0.0
            if keys[pygame.K_DOWN]:
                if self.acc[1] <= 0.0:
                    self.acc[1] -= self.ACC_INCREMENT
                else:
                    self.acc[1] = 0.0
            if keys[pygame.K_r]:
                # reset the object
                self.pos: np.array = np.array([0.0, 0.0], dtype=np.float64)
                self.vel: np.array = np.array([0.0, 0.0], dtype=np.float64)
                self.acc: np.array = np.array([0.0, 0.0], dtype=np.float64)
                self.last_x.extend(np.zeros(self.MEASUREMENTS))
                self.last_y.extend(np.zeros(self.MEASUREMENTS))
            if keys[pygame.K_s]:
                # stop the object
                self.vel: np.array = np.array([0.0, 0.0], dtype=np.float64)
                self.acc: np.array = np.array([0.0, 0.0], dtype=np.float64)

            # limit the acceleration
            self.acc = np.minimum(self.acc, (+self.ACC_LIMIT, +self.ACC_LIMIT))
            self.acc = np.maximum(self.acc, (-self.ACC_LIMIT, -self.ACC_LIMIT))

            # simulate process noise
            self.acc = self.acc + np.random.normal(0.0, self.PROCESS_NOISE_STD, 2)

            # simulate true motion
            self.vel += self.acc * self.DT
            self.pos += self.vel * self.DT + 0.5 * self.acc * self.DT ** 2

            # Noisy measurement
            meas = self.pos + np.random.normal(0.0, self.MEASUREMENT_NOISE_STD, 2)

            # predict the next position
            self.kf.predict()

            # correct the position with the new measurement
            estimate = self.kf.update(meas)
            estimate_p = estimate[0:2]
            estimate_v = estimate[2:4]
            estimate_a = estimate[4:6]

            # add the last measurements to the circular buffer
            self.last_x.append(meas[0])
            self.last_y.append(meas[1])

            # calculate simple filtered measurements (mean)
            mean_x = np.mean(self.last_x)
            mean_y = np.mean(self.last_y)

            # calculate simple filtered measurements (median)
            median_x = np.median(self.last_x)
            median_y = np.median(self.last_y)

            # real position: Blue
            pygame.draw.circle(self.screen, (0, 0, 255), self.to_screen(self.pos), 6)

            # noisy measurement: Red
            pygame.draw.circle(self.screen, (255, 0, 0), self.to_screen(meas), 2)

            # estimate mean: Gray
            pygame.draw.circle(self.screen, (32, 32, 32), self.to_screen((mean_x, mean_y)), 2)

            # estimate median: Gray
            pygame.draw.circle(self.screen, (32, 32, 32), self.to_screen((median_x, median_y)), 2)

            # estimated by the Kalman filter (position point): Green
            pygame.draw.circle(self.screen, (0, 255, 0), self.to_screen(estimate_p.ravel()), 4)

            # estimated by the Kalman filter (speed vector): Green
            pygame.draw.line(
                self.screen, (0, 255, 0),
                self.to_screen(estimate_p.ravel()),
                self.to_screen((estimate_p + estimate_v).ravel()),
                2,
            )

            # parameters
            self.screen.blit(self.font.render(f'px: {self.pos[0]:.2f}', True, (0, 0, 0)), (5, 10))
            self.screen.blit(self.font.render(f'py: {self.pos[1]:.2f}', True, (0, 0, 0)), (5, 40))
            self.screen.blit(self.font.render(f'vx: {self.vel[0]:.2f}', True, (0, 0, 0)), (5, 70))
            self.screen.blit(self.font.render(f'vy: {self.vel[1]:.2f}', True, (0, 0, 0)), (5, 100))
            self.screen.blit(self.font.render(f'ax: {self.acc[0]:.2f}', True, (0, 0, 0)), (5, 130))
            self.screen.blit(self.font.render(f'ay: {self.acc[1]:.2f}', True, (0, 0, 0)), (5, 160))

            pygame.display.flip()

            self.clock.tick(self.FPS)

            if keys[pygame.K_p]:
                pygame.image.save(self.screen, '/tmp/simulator.png')
            if keys[pygame.K_ESCAPE]:
                running = False

        pygame.quit()

    @classmethod
    def to_screen(cls, p):
        return int(cls.WIDTH / 2.0 + p[0] * cls.SCALE), int(cls.HEIGHT / 2.0  - p[1] * cls.SCALE)


if __name__ == '__main__':
    simulator = Simulator()
    simulator.loop()
