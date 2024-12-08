import numpy as np
import math

from pacman_module.game import Agent, Directions, manhattanDistance
from pacman_module.util import PriorityQueue


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, ni, nj)
            of T_t is the probability P(X_t = (ni, nj) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (ni, nj).
        """

        # Dimensions of the grid
        height, width = walls.height, walls.width

        # Initialize the transition matrix with zeros
        T_t = np.zeros((width, height, width, height))

        # Define the possible directions the ghost can move
        # 0: up, 1: down, 2: right, 3: left
        directions = ((0, 1), (0, -1), (1, 0), (-1, 0))

        for i in range(width):
            for j in range(height):
                # Consider the current position of the ghost at (i,j)

                if walls[i][j]:
                    continue
                # Valid next positions for the ghost
                valid_moves = []
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if (0 <= ni < width and 0 <= nj < height and
                            not walls[ni][nj]):
                        valid_moves.append((ni, nj))

                # Uniform distribution over valid moves
                if valid_moves:
                    if self.ghost == 'afraid':
                        prob = 1.0 / len(valid_moves)
                        adjustedProbs = []

                        for ni, nj in valid_moves:
                            # Calculate the Manhattan distance between the
                            # ghost's next position and Pacman
                            distance = manhattanDistance(position,
                                                         (ni, nj))
                            adjustedProbs.append(prob * distance)

                        # Normalize probabilities
                        NormalizedProbs = sum(adjustedProbs)

                        # To prevent division by zero
                        if NormalizedProbs > 0:
                            adjustedProbs = [p / NormalizedProbs
                                             for p in adjustedProbs]

                        # Assign normalized probabilities to the
                        # transition matrix
                        for idx, (ni, nj) in enumerate(valid_moves):
                            T_t[i, j, ni, nj] = adjustedProbs[idx]

                    elif self.ghost == 'fearless':
                        # Assign equal probabilities to all valid moves
                        prob = 1.0 / len(valid_moves)
                        for ni, nj in valid_moves:
                            T_t[i, j, ni, nj] = prob

                    elif self.ghost == 'terrified':
                        prob = 1.0 / len(valid_moves)
                        adjustedProbs = []

                        for ni, nj in valid_moves:
                            # Calculate the Manhattan distance between the
                            # ghost's next position and Pacman
                            distance = manhattanDistance(position, (ni, nj))
                            adjustedProbs.append(prob * distance)

                        # Find the maximum adjusted probability
                        max_prob = max(adjustedProbs)

                        # Find all indices corresponding
                        # to the maximum probability
                        max_indices = [idx for idx, p in
                                       enumerate(adjustedProbs)
                                       if p == max_prob]

                        # Assign probabilities
                        for idx, (ni, nj) in enumerate(valid_moves):
                            if idx in max_indices:
                                # Divide equally among max indices
                                T_t[i, j, ni, nj] = 1.0 / len(max_indices)
                            else:
                                # Assign zero probability
                                T_t[i, j, ni, nj] = 0.0
        return T_t

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """

        # Dimensions of the grid
        height, width = walls.height, walls.width

        # Initialize the observation matrix with zeros
        O_t = np.zeros((width, height))
        p = 0.5
        n = 4

        for i in range(width):
            for j in range(height):

                # Consider the current position of the ghost at (i,j)
                # Skip walls
                if walls[i][j]:
                    continue

                # Calculate the Manhattan distance between the ghost's
                distance = manhattanDistance(position, (i, j))

                # Calculate the adjusted distance
                adjustedDist = evidence - distance + n * p

                # Skip negative distances
                if adjustedDist < 0:
                    continue

                # Calculate the probability of the evidence given the
                # adjusted distance
                O_t[i, j] = math.comb(n, int(adjustedDist)) * (
                            (p**adjustedDist) * ((1 - p)**(n-adjustedDist)))

        return O_t

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """

        Trans = self.transition_matrix(walls, position)
        Obs = self.observation_matrix(walls, evidence, position)

        # update the transition matrix with the belief with values in [0, 1]
        transitionBelief = np.tensordot(belief, Trans, axes=([0, 1], [0, 1]))

        # update the belief with the observation matrix
        newBelief = np.multiply(transitionBelief, Obs)

        # Normalize the belief
        NormalizedBelief = np.sum(newBelief)

        # If the sum of the belief is 0, create a uniform belief over all
        if NormalizedBelief == 0:
            # Create a uniform belief over all non-wall cells
            newBelief = np.ones_like(belief)

            # Set walls to 0 probability
            for i in range(belief.shape[0]):
                for j in range(belief.shape[1]):
                    if walls[i][j]:  # Check if the cell is a wall
                        newBelief[i, j] = 0

            # Normalize the uniform belief matrix
            NormalizedBelief = np.sum(newBelief)

            if NormalizedBelief == 0:
                # If for some reason everything is a wall or there was no
                # valid space, set the belief to a uniform distribution
                newBelief = np.ones_like(belief)

                # Distribute evenly over all cells (including walls)
                newBelief /= newBelief.size

            else:
                # Normalize the belief
                newBelief /= NormalizedBelief
        else:
            # Normalize the belief
            newBelief /= NormalizedBelief

        return newBelief

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

    def inBounds(self, position, walls):
        """
        Check if a given position is within the bounds
        of the grid and not a wall.

        Args:
            position (tuple): A tuple (x, y) representing
                            the position to check.
            walls (object): An object representing the grid with attributes
                            'width' and 'height', and supports indexing to
                            check if a position is a wall.

        Returns:
            bool: True if the position is within bounds and not a wall,
                False otherwise.
        """
        if (position[0] < 0 or
            position[0] >= walls.width or
            position[1] < 0 or
            position[1] >= walls.height or
                walls[position[0]][position[1]]):
            return False
        return True

    def a_star_with_heuristic(self, start, target, walls):
        """
        A* search algorithm to find the shortest path from start to target.

        Arguments:
            start: Tuple (x, y) of Pacman's starting position.
            target: Tuple (x, y) of the goal position.
            walls: A grid of walls (as a 2D array or similar structure).

        Returns:
            A list of positions [(x, y), ...] representing the path
            from start to target.
        """

        # Priority queue for A*
        fringe = PriorityQueue()

        # Start with priority 0 and an empty path
        fringe.push((start, []), 0)

        # Set of closed states
        closed = set()

        # Movement directions (dx, dy)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while not fringe.isEmpty():
            # Remove node with the lowest priority
            priority, (current, path) = fringe.pop()

            # Check if goal is reached
            if current == target:
                return path

            if current in closed:
                continue
            closed.add(current)

            # Expand successors
            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)

                # Check if the next position is valid
                if (self.inBounds(next_pos, walls)):
                    # Add to the fringe
                    new_path = path + [next_pos]

                    # Cost so far
                    g_cost = len(new_path)
                    h_cost = manhattanDistance(next_pos, target)
                    priority = g_cost + h_cost
                    fringe.push((next_pos, new_path), priority)

        # Return an empty path if no solution is found
        return []

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        ghosts = []

        # Find the center of mass for each ghost belief state
        for belief, is_eaten in zip(beliefs, eaten):
            if is_eaten:
                continue

            # Compute the center of mass for the ghost belief state
            total_mass = np.sum(belief)
            if total_mass == 0:
                continue

            center_of_mass = np.array([0.0, 0.0])
            for i in range(belief.shape[0]):
                for j in range(belief.shape[1]):
                    center_of_mass[0] += i * belief[i, j]
                    center_of_mass[1] += j * belief[i, j]

            center_of_mass /= total_mass

            # Add the center of mass to the list
            ghosts.append(center_of_mass)

        if not ghosts:
            # No ghosts detected
            return Directions.STOP

        # Find the closest ghost center
        target = min(ghosts, key=lambda
                     center: manhattanDistance(position, tuple(center)))

        target = tuple(map(round, target))

        # Plan a path using A* algorithm
        path = self.a_star_with_heuristic(position, target, walls)

        if path:
            next_position = path[0]
            deltaX = next_position[0] - position[0]
            deltaY = next_position[1] - position[1]

            if [deltaX, deltaY] == [-1, 0]:
                return Directions.WEST

            elif [deltaX, deltaY] == [1, 0]:
                return Directions.EAST

            elif [deltaX, deltaY] == [0, -1]:
                return Directions.SOUTH

            elif [deltaX, deltaY] == [0, 1]:
                return Directions.NORTH

        return Directions.STOP

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
