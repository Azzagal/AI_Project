import numpy as np
import math

from pacman_module.game import Agent, Directions, manhattanDistance


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
                    match self.ghost:
                        case 'afraid':
                            prob = 1.0 / len(valid_moves)
                            adjustedProbs = []

                            for ni, nj in valid_moves:
                                # Calculate the Manhattan distance between the
                                # ghost's next position and Pacman
                                distance = manhattanDistance(position,
                                                             (ni, nj))
                                adjustedProbs.append(prob * distance)

                            # Normalize probabilities
                            total = sum(adjustedProbs)
                            if total > 0:  # To prevent division by zero
                                adjustedProbs = [p / total
                                                 for p in adjustedProbs]

                            # Assign normalized probabilities to the
                            # transition matrix
                            for idx, (ni, nj) in enumerate(valid_moves):
                                T_t[i, j, ni, nj] = adjustedProbs[idx]

                        case 'fearless':
                            prob = 1.0 / len(valid_moves)
                            for ni, nj in valid_moves:
                                T_t[i, j, ni, nj] = prob

                        case 'terrified':
                            prob = 1.0 / len(valid_moves)
                            adjustedProbs = []

                            for ni, nj in valid_moves:
                                # Calculate the Manhattan distance between the
                                # ghost's next position and Pacman
                                distance = manhattanDistance(position,
                                                             (ni, nj))
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
                # Compute the binomial distribution
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

        T = self.transition_matrix(walls, position)
        O = self.observation_matrix(walls, evidence, position)

        # update the transition matrix with the belief with values in [0, 1]
        tansitionBelief = np.tensordot(belief, T, axes=([0, 1], [0, 1]))
        # update the belief with the observation matrix
        newBelief = np.multiply(tansitionBelief, O)
        # Normalize the belief
        newBelief = newBelief / np.sum(newBelief)

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

        # faire astar avec centre de masse des fantomes et faire un switch des fantomes
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
