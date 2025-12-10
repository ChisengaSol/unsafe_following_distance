from collections import deque
import numpy as np
import config

class VehicleState:
    def __init__(self, track_id):
        self.track_id = track_id
        # History of (x_meter, y_meter) positions
        self.history = deque(maxlen=50) 
        self.velocity = 0.0  # m/s
    
    def update(self, pos_m, dt):
        """
        Update position and calculate velocity using smoothed tracking.
        """
        if len(self.history) > 0:
            prev_x, prev_y = self.history[-1]
            curr_x, curr_y = pos_m
            
            # Displacement
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            
            # Euclidean distance
            dist = np.sqrt(dx**2 + dy**2)
            
            # Instantaneous velocity (m/s)
            inst_vel = dist / dt if dt > 0 else 0
            
            # Smoothing (Exponential Moving Average)
            alpha = 0.1
            self.velocity = (alpha * inst_vel) + ((1 - alpha) * self.velocity)
            
        self.history.append(pos_m)

def check_lane_alignment(leader_hist, follower_pos_m):
    """
    Check if Follower aligns with Leader's PAST trajectory.
    """
    if not leader_hist:
        return False
        
    follower_x, _ = follower_pos_m
    
    # Compare follower X to the average X of the leader's recent history
    recent_history = list(leader_hist)[-20:]
    avg_leader_x = sum([p[0] for p in recent_history]) / len(recent_history)
    
    lateral_dist = abs(follower_x - avg_leader_x)
    return lateral_dist < config.LANE_ALIGNMENT_THRESHOLD_X