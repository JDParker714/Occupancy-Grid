from cozmo_fsm import *
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

from BuildData import AddTrainingSetImage
from kNN_Cozmo import Get_KNN_IsFloor
from kNN_Cozmo import Train_KNN

#JD User  Table #7 wood tabble
#9 paper
#4 blue yoga mat

#Megan User  Table #3

#Path to data from collected images
data_path = 'data/occupancy_v2.csv'

class DriveKNN(StateMachineProgram):
    #plt.switch_backend('Agg')
    def __init__(self):
        super().__init__(viewer_crosshairs=True, cam_viewer=True, particle_viewer=True)

    def start(self):
        super().start()
        global data_path
        if(not os.path.isfile(data_path)):
            print("Error Data File not found")
            return

        self.KNN, self.Adapt, self.Adapt_sd = Train_KNN(data_path)
        robot.camera.color_image_enabled = True
        self.occ_size = 100.0 #cm
        self.occupancy = np.zeros((int(self.occ_size),int(self.occ_size)))
        self.occupancy_img = np.ones((int(self.occ_size),int(self.occ_size),3))
        self.cur_i = -1
        self.cur_j = -1
        
        self.fig, self.ax = plt.subplots(1,1)
        self.im = self.ax.imshow(self.occupancy_img, interpolation='none', aspect='equal', origin='lower')
        self.ax.set_xlim(100, 0)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel('Y Pos (cm)')
        self.ax.set_title('Occupancy Grid')
        self.ax.set_ylabel('X Pos (cm)')
        #cv2.destroyWindow('Occupancy')
        #cv2.namedWindow( 'Occupancy',cv2.WINDOW_NORMAL);
        #cv2.resizeWindow('Occupancy', 500, 500)
        #cv2.imshow('Occupancy',self.occupancy_img)
        #cv2.waitKey(1)
        #Create_Occupancy_Plot()

    class ProjectToGround(StateNode):
        def start(self,event=None):
            super().start(event)
            if isinstance(event, DataEvent):
                #print("Recieved\n")
                id = event.data
            else:
                id = 0

            camera_center = (320/2, 240/2)
            point = self.robot.kine.project_to_ground(*camera_center)
            #print('Camera center point is at %4.1f, %4.1f' % (point[0], point[1]))
            world_pos = self.robot.kine.base_to_joint('world').dot(point)
            world_pos[0] += self.robot.pose.position.x
            world_pos[1] += self.robot.pose.position.y
            world_pos[2] += self.robot.pose.position.z
            #tprint(world_pos)
            #print("\n")
            x = world_pos[0]
            y = world_pos[1]

            i = int((x/10.0)+(self.parent.occ_size/2.0))
            j = int((y/10.0)+(self.parent.occ_size/2.0))
            if i>=0 & i<int(self.parent.occ_size) & j>0 & j<int(self.parent.occ_size):
                self.parent.occupancy[i,j] = id

                #color
                if(id==-1): self.parent.occupancy_img[i,j,:] = [100.0/255.0,100.0/255.0,100.0/255.0]
                elif(id==4): self.parent.occupancy_img[i,j,:] = [0,0,1]
                elif(id==7): self.parent.occupancy_img[i,j,:] = [0,1,0]
                else:  self.parent.occupancy_img[i,j,:] = [1,0,0]

                print("i: ", i, " j: ", j, " id: ", id, "\n")

                self.parent.im.set_data(self.parent.occupancy_img)
                self.parent.fig.canvas.draw_idle()
                plt.pause(0.01)
                #cv2.imshow('Occupancy',self.parent.occupancy_img)
                #cv2.waitKey(1)
            self.post_completion

    class GrabPatch(StateNode):
        def start(self,event=None):
            print("grabbing")
            super().start(event)
            img = np.array(self.robot.world.latest_image.raw_image)

            # Boost green to compensate for Cozmo camera idiosyncracies
            img[:,:,1] = np.minimum(231,img[:,:,1]) * 1.10

            self.parent.patch = img[105:135, 145:175, :]
            patch2 = cv2.cvtColor(self.parent.patch, cv2.COLOR_RGB2BGR)
            id = Get_KNN_IsFloor(patch2,self.parent.KNN,self.parent.Adapt, self.parent.Adapt_sd)
            self.post_data(id)

    def setup(self):
        """
            #dispatch: StateNode()
            #dispatch =TM('p')=> project
            #dispatch =TM('g')=> grab
    
            project: SetHeadAngle(-25) =T(1)=> grab
            #grab: self.GrabPatch() =N=> dispatch
    
            grab: self.GrabPatch() =D=> self.ProjectToGround()=T(2) => grab
    
            forward:  Forward(20) =T(2)=> patchwork_f
            patchwork_f: self.GrabPatch() =D=> self.ProjectToGround()=T(2)=> turn
    
            turn:  Turn(-10) =C=> forward
        """
        
        # Code generated by genfsm on Fri May  8 13:52:25 2020:
        
        project = SetHeadAngle(-25) .set_name("project") .set_parent(self)
        grab = self.GrabPatch() .set_name("grab") .set_parent(self)
        projecttoground1 = self.ProjectToGround() .set_name("projecttoground1") .set_parent(self)
        forward = Forward(20) .set_name("forward") .set_parent(self)
        patchwork_f = self.GrabPatch() .set_name("patchwork_f") .set_parent(self)
        projecttoground2 = self.ProjectToGround() .set_name("projecttoground2") .set_parent(self)
        turn = Turn(-10) .set_name("turn") .set_parent(self)
        
        timertrans1 = TimerTrans(1) .set_name("timertrans1")
        timertrans1 .add_sources(project) .add_destinations(grab)
        
        datatrans1 = DataTrans() .set_name("datatrans1")
        datatrans1 .add_sources(grab) .add_destinations(projecttoground1)
        
        timertrans2 = TimerTrans(2) .set_name("timertrans2")
        timertrans2 .add_sources(projecttoground1) .add_destinations(grab)
        
        timertrans3 = TimerTrans(2) .set_name("timertrans3")
        timertrans3 .add_sources(forward) .add_destinations(patchwork_f)
        
        datatrans2 = DataTrans() .set_name("datatrans2")
        datatrans2 .add_sources(patchwork_f) .add_destinations(projecttoground2)
        
        timertrans4 = TimerTrans(2) .set_name("timertrans4")
        timertrans4 .add_sources(projecttoground2) .add_destinations(turn)
        
        completiontrans1 = CompletionTrans() .set_name("completiontrans1")
        completiontrans1 .add_sources(turn) .add_destinations(forward)
        
        return self


  # now we are thinking of doing k-nearest neighbors, or k means clustering
    # want to get the hue pixels of our images (eventually, filter responses) and send into KMeans
    # to do: get the hue pixels, create the histogram

