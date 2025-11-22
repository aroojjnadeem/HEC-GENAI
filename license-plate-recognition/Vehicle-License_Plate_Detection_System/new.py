import os
import ast
import csv
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import torch

try:
    import ultralytics.nn.tasks as tasks
    torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': tasks.DetectionModel})
except (ImportError, AttributeError):
    print("Could not add ultralytics.nn.tasks.DetectionModel to safe globals. This might be fine for newer YOLO versions.")

from ultralytics import YOLO
import easyocr
import string

from filterpy.kalman import KalmanFilter

def linear_assignment(cost_matrix):
    """
    Solves the linear assignment problem.
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    """
    Computes Intersection over Union (IoU) between two sets of bounding boxes.
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) +
              (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top-left and x2,y2 is the bottom-right.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    """
    Sort is a simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) 
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character correction
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def license_complies_format(text):
    """
    Check if the license plate text complies with the expected format.
    Format: 2 letters, 2 numbers, 3 letters
    """
    if len(text) != 7:
        return False
    
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False

def format_license(text):
    """
    Format the license plate text by correcting characters based on predefined mappings.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 
               4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given crop using EasyOCR.
    """
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

def get_car(license_plate, vehicle_track_ids):
    """
    Check if a license plate's center is inside a tracked vehicle's bounding box.
    This is a more robust method than checking for full containment.
    """
    # Get the coordinates of the license plate
    x1, y1, x2, y2, score, class_id = license_plate
    
    # Calculate the center point of the license plate
    lp_center_x = (x1 + x2) / 2
    lp_center_y = (y1 + y2) / 2
    
    # Iterate through all tracked vehicles
    for vehicle_track in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track
        
        # Check if the license plate's center is within the vehicle's bounding box
        if xcar1 < lp_center_x < xcar2 and ycar1 < lp_center_y < ycar2:
            # If it is, we have found the car for this license plate
            return vehicle_track

    # If no matching car is found, return -1s
    return -1, -1, -1, -1, -1


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """
    Draw a border around a bounding box.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

def process_video_and_detect(video_path):
    """
    Processes the video, detects vehicles and license plates, and returns raw detection data.
    """
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')
    mot_tracker = Sort()
    vehicles = [2, 3, 5, 7] # car, motorcycle, bus, truck
    
    cap = cv2.VideoCapture(video_path)
    frame_nmr = -1
    all_results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_nmr += 1

        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Ensure the tracker gets a correctly shaped array even if detections_ is empty
        if detections_:
            detections_np = np.asarray(detections_)
        else:
            detections_np = np.empty((0, 5))
        
        track_ids = mot_tracker.update(detections_np)

        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    all_results.append({
                        'frame_nmr': frame_nmr,
                        'car_id': int(car_id),
                        'car_bbox': [xcar1, ycar1, xcar2, ycar2],
                        'license_plate_bbox': [x1, y1, x2, y2],
                        'license_plate_bbox_score': score,
                        'license_number': license_plate_text,
                        'license_number_score': license_plate_text_score,
                    })
    cap.release()
    return all_results

def interpolate_missing_data(data):
    """
    Interpolates missing bounding boxes using linear interpolation.
    """
    if not data:
        return []
        
    df = pd.DataFrame(data)
    interpolated_rows = []
    unique_car_ids = df['car_id'].unique()
    
    for car_id in unique_car_ids:
        car_df = df[df['car_id'] == car_id].copy()
        
        # Sort by score to prioritize the most confident detection when duplicates exist.
        if 'license_number_score' in car_df.columns:
            car_df = car_df.sort_values(by=['frame_nmr', 'license_number_score'], ascending=[True, False])
        
        # Keep only the first occurrence for each frame_nmr (which is now the highest scoring one).
        car_df.drop_duplicates(subset=['frame_nmr'], keep='first', inplace=True)
        
        car_df[['car_x1', 'car_y1', 'car_x2', 'car_y2']] = car_df['car_bbox'].apply(lambda x: pd.Series(x))
        car_df[['lp_x1', 'lp_y1', 'lp_x2', 'lp_y2']] = car_df['license_plate_bbox'].apply(lambda x: pd.Series(x))
        
        car_df.set_index('frame_nmr', inplace=True)
        
        min_frame = car_df.index.min()
        max_frame = car_df.index.max()
        full_index = pd.Index(range(int(min_frame), int(max_frame) + 1), name='frame_nmr')
        car_df = car_df.reindex(full_index)

        car_df[['car_x1', 'car_y1', 'car_x2', 'car_y2']] = car_df[['car_x1', 'car_y1', 'car_x2', 'car_y2']].interpolate(method='linear')
        car_df[['lp_x1', 'lp_y1', 'lp_x2', 'lp_y2']] = car_df[['lp_x1', 'lp_y1', 'lp_x2', 'lp_y2']].interpolate(method='linear')
        
        car_df[['car_id', 'license_number', 'license_number_score']] = car_df[['car_id', 'license_number', 'license_number_score']].ffill().bfill()
        
        car_df['car_bbox'] = car_df[['car_x1', 'car_y1', 'car_x2', 'car_y2']].values.tolist()
        car_df['license_plate_bbox'] = car_df[['lp_x1', 'lp_y1', 'lp_x2', 'lp_y2']].values.tolist()

        interpolated_rows.append(car_df.reset_index().to_dict('records'))
    
    final_data = [item for sublist in interpolated_rows for item in sublist]
    return final_data

def visualize_and_save_video(video_path, final_data, output_path='out.mp4'):
    """
    Visualizes the results on the video and saves it to a file.
    """
    if not final_data:
        print("No data to visualize.")
        return

    results_df = pd.DataFrame(final_data)
    results_df.set_index('frame_nmr', inplace=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    license_plate_visuals = {}
    for car_id in results_df['car_id'].unique():
        car_df = results_df[results_df['car_id'] == car_id]
        if 'license_number_score' in car_df.columns and not car_df['license_number_score'].isnull().all():
            max_score_row = car_df.loc[car_df['license_number_score'].idxmax()]
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, max_score_row.name)
            ret, frame_for_crop = cap.read()
            if not ret: continue
            
            lp_bbox = max_score_row['license_plate_bbox']
            x1, y1, x2, y2 = [int(coord) for coord in lp_bbox]
            license_crop = frame_for_crop[y1:y2, x1:x2, :]

            if license_crop.size > 0:
                # Resize for display
                target_height = 100 # A larger height for better visibility
                aspect_ratio = license_crop.shape[1] / license_crop.shape[0]
                target_width = int(target_height * aspect_ratio)
                license_crop_resized = cv2.resize(license_crop, (target_width, target_height))
                
                license_plate_visuals[car_id] = {
                    'license_crop': license_crop_resized,
                    'license_plate_number': max_score_row['license_number']
                }
                
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_nmr = -1
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream detected.")
                break
            
            frame_nmr += 1
            
            if frame_nmr in results_df.index:
                df_ = results_df.loc[[frame_nmr]]

                for _, row in df_.iterrows():
                    car_bbox = row.get('car_bbox')
                    lp_bbox = row.get('license_plate_bbox')
                    car_id = row.get('car_id')

                    if car_bbox and all(np.isfinite(car_bbox)):
                        car_x1, car_y1, car_x2, car_y2 = [int(c) for c in car_bbox]
                        draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 10, 100, 100)

                    if lp_bbox and all(np.isfinite(lp_bbox)):
                        x1, y1, x2, y2 = [int(c) for c in lp_bbox]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)

                    if car_id in license_plate_visuals:
                        visuals = license_plate_visuals[car_id]
                        license_crop = visuals['license_crop']
                        lp_text = visuals['license_plate_number']
                        
                        H, W, _ = license_crop.shape
                        
                        # Position for the license plate crop and text
                        display_x1 = int(car_x1)
                        display_y1 = int(car_y1 - H - 10) # 10px padding

                        if display_y1 > 0 and (display_x1 + W) < width:
                          try:
                            # White background for text
                            cv2.rectangle(frame, (display_x1, display_y1 - 40), (display_x1 + W, display_y1), (255, 255, 255), -1)
                            # Put text
                            cv2.putText(frame, lp_text, (display_x1 + 5, display_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            # Put license plate image
                            frame[display_y1:display_y1 + H, display_x1:display_x1 + W] = license_crop
                          except Exception as e:
                            # This can fail if the ROI is out of bounds, just skip drawing
                            pass

            out.write(frame)
            if frame_nmr % 100 == 0:
                print(f"Processed and wrote frame {frame_nmr} to video.")

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt: Terminating video writing process.")
        
    finally:
        print("Releasing video writer and capture objects.")
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Objects released successfully. Check for the output video file at {output_path}")

def main():
    """
    Main function to run the entire pipeline from start to finish.
    """
    # --- IMPORTANT: Update these paths to your local files ---
    video_path = './sample.mp4'
    output_video_path = './out.mp4'

    # Check if input video exists
    if not os.path.exists(video_path):
        print(f"Error: Input video not found at '{video_path}'")
        print("Please download the sample video or update the path.")
        return

    print("Step 1: Processing video, detecting cars and license plates...")
    raw_detections = process_video_and_detect(video_path)
    
    if not raw_detections:
        print("No license plates were successfully associated with a vehicle. Exiting.")
        return

    print(f"Step 1 finished. Found {len(raw_detections)} raw detections.")
    
    print("Step 2: Interpolating missing data points...")
    final_data = interpolate_missing_data(raw_detections)
    print(f"Step 2 finished. Total data points after interpolation: {len(final_data)}")
    
    print("Step 3: Creating and saving the output video...")
    visualize_and_save_video(video_path, final_data, output_video_path)
    
    print(f"\nPipeline completed successfully. Output video saved to {output_video_path}")

if __name__ == '__main__':
    main()