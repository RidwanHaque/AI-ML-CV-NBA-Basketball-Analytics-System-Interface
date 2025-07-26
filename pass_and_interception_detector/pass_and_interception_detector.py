from copy import deepcopy

class PassAndInterceptionDetector():
    """
    A class that detects passes between teammates and interceptions by opposing teams.
    """
    def __init__(self):
        pass 

    def detect_passes(self,ball_acquisition,player_assignment):
        """
        Detects successful passes between players of the same team.

        Args:
            ball_acquisition (list): A list indicating which player has possession of the ball in each frame.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.

        Returns:
            list: A list where each element indicates if a pass occurred in that frame
                (-1: no pass, 1: Team 1 pass, 2: Team 2 pass).
        """
        
        passes = [-1] * len(ball_acquisition)
        prev_holder=-1
        previous_frame=-1

        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:
                prev_holder = ball_acquisition[frame - 1]
                previous_frame= frame - 1
            
            current_holder = ball_acquisition[frame]
            
            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_assignment[previous_frame].get(prev_holder, -1)
                current_team = player_assignment[frame].get(current_holder, -1)

                if prev_team == current_team and prev_team != -1:
                    passes[frame] = prev_team

        return passes

    def detect_interceptions(self,ball_acquisition,player_assignment):
        """
        Detects interceptions where the ball possession changes between opposing teams.

        Args:
            ball_acquisition (list): A list indicating which player has possession of the ball in each frame.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.

        Returns:
            list: A list where each element indicates if an interception occurred in that frame
                (-1: no interception, 1: Team 1 interception, 2: Team 2 interception).
        """
        interceptions = [-1] * len(ball_acquisition)
        prev_holder=-1
        previous_frame=-1
        
        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:
                prev_holder = ball_acquisition[frame - 1]
                previous_frame= frame - 1

            current_holder = ball_acquisition[frame]
            
            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_assignment[previous_frame].get(prev_holder, -1)
                current_team = player_assignment[frame].get(current_holder, -1)
                
                if prev_team != current_team and prev_team != -1 and current_team != -1:
                    interceptions[frame] = current_team
        
        return interceptions