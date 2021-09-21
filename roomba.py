# Program to simulate a roomba
# Part of SciComp emerald challenge
import numpy as np
import matplotlib.pyplot as plt


class rectangularRoom:
    def __init__(self, length, breadth, n, m):
        """
        Create room using m x n tiles, using the convention:
        0: uncleaned, 1: cleaned, 2: inaccessible part of room
    
        :param m: number of rows of tiles
        :param n: number of columns of tiles
        :param length: length of the room
        :param breadth: breadth of the room
        :return: matrix of room
        """
        self.m = m
        self.n = n
        self.grid = np.zeros((m, n))
        self.length = length
        self.breadth = breadth
        self.tol = 0.2
        self.x = np.linspace(0, self.length, num=self.n + 1)
        self.y = np.linspace(0, self.breadth, num=self.m + 1)
        self.nan_count = 0

    def cordon_off_edge(self):
        """
        Cordon off the corner of the rectangular grid using NaNs.
        """
        for ii in range(self.m):
            for jj in range(self.n):
                if 0 <= self.x[jj] <= 4:
                    if (self.x[jj] + 8) <= self.y[ii] <= self.breadth:
                        self.grid[ii, jj] = np.nan
                        self.nan_count += 1

    def is_tile_inside_room(self, pos):
        """
        Returns true if tile is within the room boundaries, with some tolerance to account for finite size of the robot

        :param pos: position
        :return: T/F
        """
        if self.tol <= pos[0] <= self.length - self.tol and self.tol <= pos[1] <= self.breadth - self.tol:
            if 0 <= pos[0] <= 4:
                if (pos[0] + 8 - self.tol* np.sqrt(2)) < pos[1] <= self.breadth:
                    return False
            return True
        return False

    def get_random_location(self):
        """"
        :return: random location inside the room
        """
        pos_x = np.random.rand() * self.length
        if 0 <= pos_x <= 4:
            pos_y = np.random.rand() * (pos_x+8)
        else:
            pos_y = np.random.rand() * self.breadth
        return [pos_x, pos_y]

    def get_grid_cell_idx(self, pos):
        """
        convert spatial location to indices on grid
        """
        row = int(np.floor(pos[1]*self.m/self.breadth))
        col = int(np.floor(pos[0]*self.n/self.length))
        return [row, col]

    def mark_tile_as_cleaned(self, pos):
        """
        If robot moves to position, mark as cleaned.
        """
        grid_indices = self.get_grid_cell_idx(pos)
        # print(grid_indices, pos)
        self.grid[grid_indices[0], grid_indices[1]] = 1

    def mark_tile_idx_as_cleaned(self, idx):
        """
        Mark tile as cleaned using grid indices directly
        """
        self.grid[idx[0], idx[1]] = 1

    def get_fraction_area_cleaned(self):
        """
        Calculate fraction of total area cleaned by the robot
        """
        return np.sum(self.grid == 1) / (m * n - self.nan_count)

    def visualize(self):
        """
        Plot the pen: Green indicates square with grass, brown indicates squares which were eaten by cows
        """
        plt.pcolormesh(self.x, self.y, self.grid, cmap='Greys')

        ax = plt.gca()
        ax.set_aspect('equal')
        ax.axis('off')
        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.breadth + 0.1)
        plt.draw()
    

class Robot:
    def __init__(self, room):
        self.room = room
        self.current_pos = self.room.get_random_location()
        room.mark_tile_as_cleaned(self.current_pos)
        self.theta = (-1 + 2 * np.random.rand()) * np.pi
        self.speed = 0.8
        self.dt = 0.2
        self.tol = 0.5
        self.total_time = 0
        self.start = self.current_pos
        self.num_wall_collisions = 0

        # Mark initial location of the bot
        # plt.plot(self.current_pos[0], self.current_pos[1], 'og')

    def get_position(self):
        """
        Returns current position of the robot
        """
        return self.current_pos

    def get_direction(self):
        """
        Get orientation of the robot
        """
        return self.theta

    def reorient(self):
        """
        Change direction randomly
        """
        self.theta = -np.pi + 2 * np.pi * np.random.rand()

    def get_total_time(self):
        """
        Returns total time spent by robot so far.
        Note: Re-orientations are assumed to be instantaneous.
        """
        return self.total_time/60

    def move(self, bresenham_opt):
        """
        Move robot for 1 time-step

        :return:
        """
        n_heading = np.array([np.cos(self.theta), np.sin(self.theta)])  # normal vector along \theta
        new_position = self.current_pos + self.speed * self.dt * n_heading
        if room.is_tile_inside_room(new_position):
            # self.draw_path(new_position)

            self.current_pos = new_position
            self.total_time += self.dt
            if not bresenham_opt:
                room.mark_tile_as_cleaned(self.current_pos)
        else:
            self.num_wall_collisions += 1
            if bresenham_opt:
                self.bresenham()
                self.start = self.current_pos
            self.reorient()

    def bresenham(self):
        """
        Finds all cells of the 2D grid that are intersected by the robot path

        :return:
        """
        slope = np.tan(self.theta)

        start_idx = room.get_grid_cell_idx(self.start)
        end_idx = room.get_grid_cell_idx(self.current_pos)
        intercept = start_idx[0] - slope * start_idx[1]

        if -1 <= slope <= 1:
            low = np.amin([start_idx[1], end_idx[1]])
            high = np.amax([start_idx[1], end_idx[1]])
            for j in range(low, high):
                i = int(np.floor(slope * j + intercept))
                room.mark_tile_idx_as_cleaned([i, j])
        else:
            low = np.amin([start_idx[0], end_idx[0]])
            high = np.amax([start_idx[0], end_idx[0]])
            for i in range(low, high):
                j = int(np.floor((i - intercept)/slope))
                room.mark_tile_idx_as_cleaned([i, j])

    def draw_path(self, pos):
        """
        Draw the last step taken by the robot

        :param new_pos:
        """
        # plt.plot([self.current_pos[0], pos[0]], [self.current_pos[1], pos[1]], '-ob', markersize=2)
        plt.plot([self.current_pos[0], pos[0]], [self.current_pos[1], pos[1]], '-r')
        plt.draw()
        # plt.pause(0.1)


if __name__ == '__main__':
    # np.random.seed(5)

    length, breadth = 10, 12
    n, m = 30, 36
    bresenham_opt = 0  # set 1 to use Bresenham line algorithm to find cleaned tiles

    # num_trials = 50
    # time_taken = np.zeros((num_trials,))
    # for trial_idx in range(num_trials):
    #     room = rectangularRoom(length, breadth, n, m)
    #     room.cordon_off_edge()
    #     roomba = Robot(room)
    #     while room.get_fraction_area_cleaned() < 0.95:
    #         roomba.move(bresenham_opt)
    #         if bresenham_opt:
    #             roomba.bresenham()
    #     time_taken[trial_idx] = roomba.get_total_time()
    #     print("time taken= ", time_taken[trial_idx])
    # print("Avg. time for cleaning 95% area= ", np.mean(time_taken))

    num_trials = 10
    frac_cleaned = np.zeros((num_trials,))
    for trial_idx in range(num_trials):
        room = rectangularRoom(length, breadth, n, m)
        room.cordon_off_edge()
        roomba = Robot(room)
        while roomba.get_total_time() < 10:
            roomba.move(bresenham_opt)
            # roomba.bresenham()
        frac_cleaned[trial_idx] = room.get_fraction_area_cleaned()
    print("Fraction of Area cleaned (percent) = %2.2f" % np.mean(frac_cleaned))
    # print("Total time (mins)= %2.2f" % roomba.get_total_time())

    # room.visualize()
    #
    # plt.plot([0, length], [0, 0], c='k')
    # plt.plot([length, length], [0, breadth], 'k')
    # plt.plot([length, 4], [breadth, breadth], 'k')
    # plt.plot([4, 0], [breadth, 8], 'k')
    # plt.plot([0, 0], [8, 0], 'k')
    #
    # plt.show()
