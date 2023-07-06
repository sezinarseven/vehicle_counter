import cv2
from matplotlib import pyplot as plt
import numpy as np
import yaml
from yolox.tracker.basetrack import TrackState
from yolox.tracker.byte_tracker import STrack
from yolox.tracker.kalman_filter import KalmanFilter
from yolox.tracker import matching
from yolox.utils.detections import Detections
from yolox.utils.my_utils import find_class_names, joint_stracks, read_class_names, remove_duplicate_stracks, sub_stracks

# 

class v7_ByteTracker(object):
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20 = False, frame_rate=30):
        self.tracked_stracks = []  
        self.lost_stracks = []
        self.removed_stracks = []

        self.track_thresh=track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20 # test mot20
        self.class_names = read_class_names()
        self.classes_list = yaml.load(open("coco.yaml"), Loader=yaml.SafeLoader)['names']
        self.frame_id = 0
        
        self.det_thresh = self.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        cmap = plt.get_cmap('tab20b') #initialize color map
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        self.colors = colors
        self.track_id = 0


    def update(self, frame, tracker, output_results):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = output_results[:, 4]

        bboxes = output_results[:, :4]

        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]

        if len(output_results) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cls) for (*tlbr, s, cls) in output_results[:,:6]]
        else:
            detections = []


        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                self.track_id=self.track_id+1
                
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        if len(dets_second) > 0:

            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cls) for
                          (tlbr, s, cls) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []
        
        for t in output_stracks:
            output = []
            tlwh = t.tlwh
            tid = t.track_id
            tlwh = np.expand_dims(tlwh, axis=0)
            xyxy = STrack.tlwh_to_tlbr(tlwh)          
            xyxy = np.squeeze(xyxy, axis=0)
            output.extend(xyxy)
            output.append(tid)
            output.append(t.cls)
            output.append(t.score)     
            outputs.append(output)
            
        for track in outputs:  
            class_name = find_class_names(int(track[5]))
            color = tracker.colors[int(track[5]) % len(tracker.colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(track[0]), int(track[1])), ((int(track[0])+int(track[2])), (int(track[1])+int(track[3]))), color, 2)
            cv2.rectangle(frame, (int(track[0]), int(track[1]-30)), (int(track[0])+(len(class_name)+len(str(track[4])))*17, int(track[1])), color, -1)
            cv2.putText(frame, (str(class_name) + " : " + str(int(track[4]))),(int(track[0]), int(track[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
                
        return outputs