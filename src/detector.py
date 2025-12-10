import cv2
import numpy as np
from ultralytics import YOLO
import os

import config
import utils
import tracker

def main():
    #Setup Resources
    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: Model not found at {config.MODEL_PATH}")
        return

    model = YOLO(config.MODEL_PATH)
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    
    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps if fps > 0 else 0.033
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.OUTPUT_PATH, fourcc, fps, (width, height))
    
    # State Dictionary: {track_id: VehicleState}
    vehicles = {}
    
    print(f"Processing {config.VIDEO_PATH}...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        #Tracking (YOLO)
        results = model.track(frame, persist=True, verbose=False)
        
        current_frame_ids = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            
            # Map ID to Bounding Box for easy visualization later
            id_to_bbox = {id: box for id, box in zip(ids, boxes)}
            
            for track_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = box
                current_frame_ids.append(track_id)
                
                # Ground Contact Point (Bottom Center)
                cx = (x1 + x2) / 2
                cy = y2 
                
                #IPM Conversion (Pixels -> Meters)
                pos_m = utils.apply_ipm((cx, cy), frame_height=height)
                
                # Update Vehicle State
                if track_id not in vehicles:
                    vehicles[track_id] = tracker.VehicleState(track_id)
                
                vehicles[track_id].update(pos_m, dt)
                
                # Visualization: Draw Box & Speed
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id} {vehicles[track_id].velocity:.1f}m/s", 
                            (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Pairwise Analysis (Lane Alignment & TTC)
            processed_pairs = set()
            
            for id_follower in current_frame_ids:
                follower_state = vehicles[id_follower]
                
                for id_leader in current_frame_ids:
                    if id_follower == id_leader: continue
                    
                    # Prevent checking the same pair twice
                    pair_key = tuple(sorted((id_follower, id_leader)))
                    if pair_key in processed_pairs: continue
                        
                    leader_state = vehicles[id_leader]
                    
                    # Check Geometry
                    f_y_pixel = id_to_bbox[id_follower][3]
                    l_y_pixel = id_to_bbox[id_leader][3]
                    
                    if l_y_pixel >= f_y_pixel:
                        continue # Leader is actually behind or parallel
                        
                    # Check Lane Alignment
                    if tracker.check_lane_alignment(leader_state.history, follower_state.history[-1]):
                        
                        # TTC Calculation
                        f_pos = follower_state.history[-1]
                        l_pos = leader_state.history[-1]
                        
                        distance_m = np.sqrt((f_pos[0]-l_pos[0])**2 + (f_pos[1]-l_pos[1])**2)
                        
                        # Time Headway = Distance / Speed
                        if follower_state.velocity > 0.1:
                            ttc = distance_m / follower_state.velocity
                        else:
                            ttc = 99.9
                            
                        # Flag Unsafe Following
                        if ttc < config.TTC_THRESHOLD:
                            # Visualization: Red Line
                            f_box = id_to_bbox[id_follower]
                            l_box = id_to_bbox[id_leader]
                            f_center = (int((f_box[0]+f_box[2])/2), int(f_box[3]))
                            l_center = (int((l_box[0]+l_box[2])/2), int(l_box[3]))
                            
                            cv2.line(frame, f_center, l_center, (0, 0, 255), 3)
                            cv2.putText(frame, f"UNSAFE: {ttc:.1f}s", 
                                        (f_center[0], int(f_center[1] - 20)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                    processed_pairs.add(pair_key)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Done. Saved to {config.OUTPUT_PATH}")

if __name__ == "__main__":
    main()