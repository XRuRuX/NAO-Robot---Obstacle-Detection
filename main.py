import cv2
import numpy as np
import time
from naoqi import ALProxy

def stand_up(robot_ip, robot_port):
    try:
        posture_proxy = ALProxy("ALRobotPosture", robot_ip, robot_port)
    except Exception as e:
        print("Could not create proxy to ALRobotPosture")
        print("Error was: ", e)
        return

    posture_proxy.goToPosture("Stand", 1.0)

    # Additional postures if needed
    # posture_proxy.goToPosture("Crouch", 1.0)
    # posture_proxy.goToPosture("Sit", 1.0)

    print(posture_proxy.getPostureFamily())

def move_head_down(robot_ip, robot_port):
    try:
        motion_proxy = ALProxy("ALMotion", robot_ip, robot_port)
    except Exception as e:
        print("Could not create proxy to ALMotion")
        print("Error was: ", e)
        return

    # Move the head down by adjusting the pitch angle
    motion_proxy.setAngles("HeadPitch", 0.2, 0.1)  # Adjust the angle as needed

def move_forward(robot_ip, robot_port):
    try:
        motion_proxy = ALProxy("ALMotion", robot_ip, robot_port)
    except Exception as e:
        print("Could not create proxy to ALMotion")
        print("Error was: ", e)
        return

    # Move forward
    motion_proxy.move(0.05, 0.0, -0.04)  # Adjust the velocity as needed
    time.sleep(5)
    motion_proxy.stopMove()
    stopMove = False

def main(robot_ip, robot_port):
    try:
        motion_proxy = ALProxy("ALMotion", robot_ip, robot_port)
        video_proxy = ALProxy("ALVideoDevice", robot_ip, robot_port)
    except Exception as e:
        print("Connection to the robot failed: {}".format(str(e)))
        return

    stand_up(robot_ip, robot_port)
    move_head_down(robot_ip, robot_port)
    time.sleep(2)  # Allow time for the robot to stabilize after moving its head down

    resolution = 2
    color_space = 11
    fps = 10

    camera_id = video_proxy.subscribeCamera("python_client", 0, resolution, color_space, fps)
    print("Video stream opened successfully.")

    net = cv2.dnn.readNet("C:/NAOSDK/yolov3.weights", "C:/NAOSDK/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    cv2.namedWindow("NAO Camera", cv2.WINDOW_NORMAL)
    image_count = 300
    try:
        while True:
            image_data = video_proxy.getImageRemote(camera_id)
            width, height = image_data[0], image_data[1]
            image_array = np.frombuffer(image_data[6], dtype=np.uint8).reshape((height, width, 3))

            # Draw blue rectangle in the middle of the screen
            rect_width = int(width / 2)
            rect_height = int(height / 2) + 100
            rect_x = int((width - rect_width) / 2)
            rect_y = int((height - rect_height) / 2)
            cv2.rectangle(image_array, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 0, 0), 2)

            blob = cv2.dnn.blobFromImage(image_array, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
		
            stopMove = False
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    confidence = scores.max()
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
				
                        # Check if any part of the bounding box is inside the blue rectangle
                        if (
                            x < rect_x + rect_width and x + w > rect_x and
                            y < rect_y + rect_height and y + h > rect_y
                        ):
                            # If any part is inside, draw the bounding box in orange
			    stopMove = False
                            cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 165, 255), 2)
                            if y + h > rect_y + rect_height:
                                cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 0, 255), 2)
				stopMove = True;
				break
                        else:
                            # If not intersecting, draw the bounding box in green
                            cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
		            stopMove = False;

            cv2.imshow("NAO Camera", image_array)
            
	    image_count += 1
            image_filename = "image_{}.jpg".format(image_count)
            cv2.imwrite(image_filename, image_array)

	    if stopMove == False:
                move_forward(robot_ip, robot_port)		

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        video_proxy.unsubscribe(camera_id)
        cv2.destroyAllWindows()

        try:
            motion_proxy.stopMove()
            motion_proxy.rest()
        except Exception as e:
            print("Stopping the robot failed: {}".format(str(e)))

if __name__ == "__main__":
    robot_ip = "192.168.1.100"
    robot_port = 9559

    main(robot_ip, robot_port)

