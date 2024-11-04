from pacman_module.game import Agent, Directions
import math
from pacman_module.util import manhattanDistance

class PacmanAgent(Agent):
    def __init__(self):
        super().__init__()
        self.visited = {}  # Dictionary to maintain visited nodes for optimization
        self.depth_limit = 4  # Depth limit for H-Minimax search
        
    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.
        
        Arguments:
            state: GameState. See API or class `pacman.GameState`.
        
        Returns:
            A legal move as defined in `game.Directions`.
        """
        legal_moves = state.getLegalActions(0)  # 0 is Pacman's agent index
        
        if not legal_moves:
            return Directions.STOP
            
        best_score = float('-inf')
        best_action = Directions.STOP
        
        # Try each legal move and choose the one with the highest minimax value
        for action in legal_moves:
            next_state = state.generateSuccessor(0, action)
            score = self._minimax(next_state, 1, 0, float('-inf'), float('inf'))
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action
    
    def _minimax(self, state, agent_index, depth, alpha, beta):
        """
        Implementation of H-Minimax algorithm with alpha-beta pruning
        """
        # Terminal state check
        if state.isWin() or state.isLose() or depth == self.depth_limit:
            return self._evaluate(state)
            
        # Reset agent_index and increment depth when all agents have moved
        num_agents = state.getNumAgents()
        if agent_index == num_agents:
            agent_index = 0
            depth += 1
            
        # Get legal moves for current agent
        legal_moves = state.getLegalActions(agent_index)
        
        # Pacman's turn (maximizing)
        if agent_index == 0:
            max_value = float('-inf')
            for action in legal_moves:
                next_state = state.generateSuccessor(agent_index, action)
                value = self._minimax(next_state, agent_index + 1, depth, alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, max_value)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_value
            
        # Ghosts' turn (minimizing)
        else:
            min_value = float('inf')
            for action in legal_moves:
                next_state = state.generateSuccessor(agent_index, action)
                value = self._minimax(next_state, agent_index + 1, depth, alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, min_value)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_value
    
    def _evaluate(self, state):
        """
        Heuristic evaluation function for non-terminal states
        """
        # If it's a win or lose state, return appropriate value
        if state.isWin():
            return float('inf')
        if state.isLose():
            return float('-inf')
            
        # Get the current score
        score = state.getScore()
        
        # Get Pacman's position
        pacman_pos = state.getPacmanPosition()
        
        # Get food positions
        food_list = state.getFood().asList()
        
        # Get ghost positions and states
        ghost_states = state.getGhostStates()
        ghost_positions = [ghost.getPosition() for ghost in ghost_states]
        
        # Features calculation
        
        # 1. Food distance: negative of the closest food distance
        food_distances = [manhattanDistance(pacman_pos, food) for food in food_list]
        closest_food_distance = min(food_distances) if food_distances else 0
        food_feature = -closest_food_distance
        
        # 2. Ghost distance: sum of distances to ghosts (closer is worse)
        ghost_distances = [manhattanDistance(pacman_pos, ghost_pos) for ghost_pos in ghost_positions]
        ghost_feature = sum(1.0/dist if dist > 0 else float('-inf') for dist in ghost_distances)
        
        # 3. Remaining food penalty
        remaining_food_feature = -len(food_list) * 10
        
        # Combine features with weights
        final_score = score + food_feature - 20 * ghost_feature + remaining_food_feature
        
        return final_score