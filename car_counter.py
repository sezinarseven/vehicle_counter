import cv2
from app.detector.detection_yolo import Yolov7Detector
from app.tracker.byte_track import v7_ByteTracker
from table_class import Plotter


if __name__ == "__main__":
    detector =Yolov7Detector()
    tracker = v7_ByteTracker()
    plotter = Plotter(title="Car Counter", start_x=660, start_y=820, col_width=200, row_height=30, background_color=(150,150,150), opacity=0.6)
    cap = cv2.VideoCapture("car.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (width, height))

    vehicles = {}
    incoming_vehicles = []
    outgoing_vehicles = []
    frame_counter = 0
    in_bicycle, in_car, in_motorcycle, in_bus, in_truck, out_bicycle, out_car, out_motorcycle, out_bus, out_truck = 0,0,0,0,0,0,0,0,0,0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_counter = frame_counter + 1
        if not ret:
            break
        i=0
        line_y = 270
        cv2.line(frame, (0, line_y), (1920, line_y), (200, 0, 0), 3)

        predictions = detector.detect(frame)
        predictions = tracker.update(frame, tracker, output_results=predictions)

        for pred in predictions:
            if int(pred[5]) in [1, 2, 3, 5, 7]:  # 'bicycle', 'car', 'motorcycle', 'bus', 'truck'
                mid_y = (2*pred[1] + pred[3]) / 2
                
                if vehicles.get(pred[4]) is None:
                    vehicles[pred[4]] = mid_y

                else:
                    if line_y > vehicles.get(pred[4]):
                        vehicles.update({pred[4]: mid_y})
                        if line_y < vehicles.get(pred[4]):
                            if pred[4] not in incoming_vehicles:
                                incoming_vehicles.append(pred[4])
                                cv2.line(frame, (0, line_y), (1920, line_y), (0, 0, 200), 3)
                                cv2.putText(frame, "Incoming Vehicle ID: " + str(pred[4]), (70, 1000+i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (215, 160, 255), 2)
                                i=i-30
                                if int(pred[5])==1:
                                    in_bicycle+=1
                                elif int(pred[5])==2:
                                    in_car+=1
                                elif int(pred[5])==3:
                                    in_motorcycle+=1
                                elif int(pred[5])==5:
                                    in_bus+=1
                                elif int(pred[5])==7:
                                    in_truck+=1
                    
                    elif line_y < vehicles.get(pred[4]):
                        vehicles.update({pred[4]: mid_y})
                        if line_y > vehicles.get(pred[4]):
                            if pred[4] not in outgoing_vehicles:
                                outgoing_vehicles.append(pred[4])
                                cv2.line(frame, (0, line_y), (1920, line_y), (0, 0, 200), 3)
                                cv2.putText(frame, "Outgoing Vehicle ID: " + str(pred[4]), (70, 1000+i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (204, 153, 255), 2)
                                i=i-30
                                if int(pred[5])==1:
                                    out_bicycle+=1
                                elif int(pred[5])==2:
                                    out_car+=1
                                elif int(pred[5])==3:
                                    out_motorcycle+=1
                                elif int(pred[5])==5:
                                    out_bus+=1
                                elif int(pred[5])==7:
                                    out_truck+=1
        
        data = ["", "Incoming Vehicles", "Outgoing Vehicles",
                "car" , str(in_car), str(out_car),
                "truck", str(in_truck), str(out_truck),
                "bus", str(in_bus), str(out_bus),
                "motorcycle", str(in_motorcycle), str(out_motorcycle),
                "bicycle", str(in_bicycle), str(out_bicycle),
                "total", str(len(incoming_vehicles)), str(len(outgoing_vehicles))] 
        
        plotter.plot_table(frame, cell_data=data, num_rows=7, num_columns=3)
        out.write(frame)

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
