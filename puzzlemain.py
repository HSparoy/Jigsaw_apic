import robcomm
import random
from rwsuis import RWS
import asyncio
from open_gopro import WirelessGoPro, Params
import cv2
import time 
import numpy as np




ROBOT_IN_POSITION = "T_ROB1/STA/in_position"
FIRST_RUN = "T_ROB1/STA/first_run"
SECOND_RUN = "T_ROB/STA/second_run"
PUZZLEPIECE_POS = "T_ROB1/STA/PuzzlePiece"
PUZZLEPIECE_NEXT_POS = "T_ROB1/STA/PuzzlePiece_next_pos"
PUZZLEPIECE_1_POS = "T_ROB1/STA/PuzzlePiece_1_pos"
PUZZLEPIECE_2_POS = "T_ROB1/STA/PuzzlePiece_2_pos"
PUZZLEPIECE_3_POS = "T_ROB1/STA/PuzzlePiece_3_pos"
PUZZLEPIECE_4_POS = "T_ROB1/STA/PuzzlePiece_4_pos"
PUZZLEPIECE_5_POS = "T_ROB1/STA/PuzzlePiece_5_pos"
PUZZLEPIECE_6_POS = "T_ROB1/STA/PuzzlePiece_6_pos"
PUZZLEPIECE_7_POS = "T_ROB1/STA/PuzzlePiece_7_pos"
PUZZLEPIECE_8_POS = "T_ROB1/STA/PuzzlePiece_8_pos"
PUZZLEPIECE_9_POS = "T_ROB1/STA/PuzzlePiece_9_pos"
CAMERA_POS = "T_ROB1/STA/Camera_Pos"
SOLVING_POS = "T_ROB1/STA/Solving_pos"
SOLVING_1_POS = "T_ROB1/STA/Solving_1_pos"
SOLVING_2_POS = "T_ROB1/STA/Solving_2_pos"
SOLVING_3_POS = "T_ROB1/STA/Solving_3_pos"
SOLVING_5_POS = "T_ROB1/STA/Solving_5_pos"
SOLVING_6_POS = "T_ROB1/STA/Solving_6_pos"
SOLVING_7_POS = "T_ROB1/STA/Solving_7_pos"
SOLVING_8_POS = "T_ROB1/STA/Solving_8_pos"
SOLVING_9_POS = "T_ROB1/STA/Solving_9_pos"
COUNT_FIRST_RUN = "T_ROB1/STA/count_first_run"
COUNT_SECOND_RUN = "T_ROB1/STA/count_second_run"
OK_SIGNAL = "T_ROB1/STA/ok_signal"



def url_for_variable(variable) -> str:
    return f"/rw/rapid/symbol/RAPID/{variable}/data"

class RobotCom:
    IN_POSITION_ = url_for_variable(ROBOT_IN_POSITION)
    FRST_RUN = url_for_variable(FIRST_RUN)
    SCDN_RUN = url_for_variable(SECOND_RUN)

    

    # This function is called when a message is received for any of the subscribed variables
    def on_message(self, robcomm: robcomm.Robot, variable_url: str, value):
        match variable_url:
            case RobotCom.FRST_RUN:
                if value == "0":
                    print("The robot is done performing the first run")
                    return
                if value == "1":
                    print("Robot has started the task")
                    return
                if value == "2":
                    #gopro = await connect_gopro()
                    #await take_picture(gopro)
                    robcomm.set_rapid_variable(PUZZLEPIECE_POS, CAMERA_POS)
                    ft1 = robcomm.get_rapid_variable(COUNT_FIRST_RUN)
                    Count_ft1_nr1 = int(ft1) +1
                    robcomm.set_rapid_variable(PUZZLEPIECE_NEXT_POS,f"PUZZLEPIECE_{Count_ft1_nr1}_POS")
                    robcomm.set_rapid_variable(OK_SIGNAL, f"{True}")
                    robcomm.set_rapid_variable(FIRST_RUN, f"{1}")
                if value == "3":
                    ft2 = robcomm.get_rapid_variable(COUNT_FIRST_RUN)
                    Count_ft2_nr2 = int(ft2) + 1
                    robcomm.set_rapid_variable(PUZZLEPIECE_POS, f"PUZZLEPIECE_{Count_ft2_nr2}_POS")
                    robcomm.set_rapid_variable(PUZZLEPIECE_NEXT_POS, CAMERA_POS)
                    robcomm.set_rapid_variable(FIRST_RUN,f"{1}")
            case RobotCom.SCDN_RUN:
                if value == "0":
                    print("The robot is done performing the second run")
                    return
                if value == "1":
                    print("Robot has started the task")
                    return
                if value == "2":
                    robcomm.set_rapid_variable(PUZZLEPIECE_POS, PUZZLEPIECE_1_POS)
                    robcomm.set_rapid_variable(PUZZLEPIECE_NEXT_POS, SOLVING_1_POS)
                    robcomm.set_rapid_variable(SECOND_RUN,f"{1}")

                if value == "3":
                    sc1 = robcomm.get_rapid_variable(COUNT_SECOND_RUN)
                    Count_sc1_nr1 = int(sc1) + 1
                    robcomm.set_rapid_variable()
                    robcomm.set_rapid_variable(PUZZLEPIECE_POS, f"PUZZLEPIECE_{Count_sc1_nr1}_POS")
                    robcomm.set_rapid_variable(SECOND_RUN,f"{1}")


                       

            
async def connect_gopro():
    try:
        gopro = WirelessGoPro()  # Initialize the GoPro object
        await gopro.connect()    # Attempt to connect to the camera
        print("Connected to GoPro.")
    except Exception as e:
        print(f"Failed to connect to GoPro: {str(e)}")
    return gopro

async def take_picture(gopro):
    async with WirelessGoPro() as gopro:
        # Start recording a short video
        assert (await gopro.http_command.set_shutter(shutter=Params.Toggle.ENABLE)).ok
        # Stop recording
        assert (await gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)).ok

        new_media = (await gopro.http_command.get_media_list()).data
        
        # Find the most recent file (the video we just filmed)
        most_recent_file = max(new_media.files, key=lambda file: int(file.creation_timestamp))

        # Download the most recent file
        response = await gopro.http_command.download_file(camera_file=most_recent_file.filename)

        # Open the video
        cap = cv2.VideoCapture(response.data.name)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # Set the video to the first frame and read it
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            # Save the frame
            cv2.imwrite("frame.png", frame)
            print("Frame saved as puzzle.png")
        else:
            print("Error: Could not read frame.")
    



def main():
   # Create a connection to the real robot
    robot = robcomm.Robot(ip='152.94.0.37', port=443)
    # This is the virtual
    #robot = robcomm.Robot(ip='127.0.0.1', port=9933)
    # Create a local example object to handle the messages
    
    # Subscribe to the in_position and call the on_message function when a message is received

    #wrd = robot.get_rapid_variable (variable=WRD)
    #if ( wrd == 0):
    #   print(" Robot is waiting for Python to set WPW ")
    #robot.set_rapid_variable(WPW,6)
    #move_camera_above_puzzle_piece(robot)
    #gopro = await connect_gopro()
    #await take_picture(gopro)
    example = RobotCom()
    robot.subscribe([FIRST_RUN], [SECOND_RUN], example.on_message)
    

if __name__ == '__main__':
    main()



     


    
    
    
    
    

