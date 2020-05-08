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

class RoamKNN(StateMachineProgram):
    #plt.switch_backend('Agg')
    def __init__(self):
        super().__init__(viewer_crosshairs=True, cam_viewer=True)

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

                if(i <= 3 or i >= self.parent.occ_size-3 or j <= 3 or j >= self.parent.occ_size-3):
                    #too close to the edge; we want it to turn
                    self.post_data(3)

                #color
                if(id==-1): 
                    self.parent.occupancy_img[i,j,:] = [100.0/255.0,100.0/255.0,100.0/255.0]
                    self.post_data(2)
                elif(id==4): 
                    self.parent.occupancy_img[i,j,:] = [0,0,1]
                    self.post_data(1)
                elif(id==7): 
                    self.parent.occupancy_img[i,j,:] = [0,1,0]
                    self.post_data(1)
                else:  
                    self.parent.occupancy_img[i,j,:] = [1,0,0]
                    self.post_data(1)

                print("i: ", i, " j: ", j, " id: ", id, "\n")

                #cv2.imshow('Occupancy',self.parent.occupancy_img)
                #cv2.waitKey(1)
            else:
                self.post_data(3)
            #return
            #self.post_completion

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
    
    class PlotOccupancy(StateNode):
        def start(self, event=None):
            super().start(event)
            self.parent.im.set_data(self.parent.occupancy_img)
            self.parent.fig.canvas.draw_idle()
            plt.pause(0.01)


    def setup(self):
        """
            #dispatch: StateNode()
            #dispatch =TM('p')=> project
            #dispatch =TM('g')=> patchwork_f
    
            project: SetHeadAngle(-25) =T(1)=> patchwork_f
            #grab: self.GrabPatch() =N=> dispatch
    
            patchwork_f: self.GrabPatch() =D=> projectTo: self.ProjectToGround()
            projectTo =D(1)=> self.PlotOccupancy() =T(2)=> Forward(30) =C=> patchwork_f
            projectTo =D(2)=> self.PlotOccupancy() =T(2)=> Turn(20) =C=> patchwork_f
            projectTo =D(3)=> self.PlotOccupancy() =T(2)=> Turn(150) =C=> patchwork_f
    
            # forward:  Forward(20) =T(2)=> patchwork_f
            # patchwork_f: self.GrabPatch() =D=> self.ProjectToGround()=T(2)=> turn
    
            # turn:  Turn(-10) =C=> forward
        """
        
        # Code generated by genfsm on Fri May  8 14:21:12 2020:
        
        project = SetHeadAngle(-25) .set_name("project") .set_parent(self)
        patchwork_f = self.GrabPatch() .set_name("patchwork_f") .set_parent(self)
        projectTo = self.ProjectToGround() .set_name("projectTo") .set_parent(self)
        plotoccupancy1 = self.PlotOccupancy() .set_name("plotoccupancy1") .set_parent(self)
        forward1 = Forward(30) .set_name("forward1") .set_parent(self)
        plotoccupancy2 = self.PlotOccupancy() .set_name("plotoccupancy2") .set_parent(self)
        turn1 = Turn(20) .set_name("turn1") .set_parent(self)
        plotoccupancy3 = self.PlotOccupancy() .set_name("plotoccupancy3") .set_parent(self)
        turn2 = Turn(150) .set_name("turn2") .set_parent(self)
        
        timertrans1 = TimerTrans(1) .set_name("timertrans1")
        timertrans1 .add_sources(project) .add_destinations(patchwork_f)
        
        datatrans1 = DataTrans() .set_name("datatrans1")
        datatrans1 .add_sources(patchwork_f) .add_destinations(projectTo)
        
        datatrans2 = DataTrans(1) .set_name("datatrans2")
        datatrans2 .add_sources(projectTo) .add_destinations(plotoccupancy1)
        
        timertrans2 = TimerTrans(2) .set_name("timertrans2")
        timertrans2 .add_sources(plotoccupancy1) .add_destinations(forward1)
        
        completiontrans1 = CompletionTrans() .set_name("completiontrans1")
        completiontrans1 .add_sources(forward1) .add_destinations(patchwork_f)
        
        datatrans3 = DataTrans(2) .set_name("datatrans3")
        datatrans3 .add_sources(projectTo) .add_destinations(plotoccupancy2)
        
        timertrans3 = TimerTrans(2) .set_name("timertrans3")
        timertrans3 .add_sources(plotoccupancy2) .add_destinations(turn1)
        
        completiontrans2 = CompletionTrans() .set_name("completiontrans2")
        completiontrans2 .add_sources(turn1) .add_destinations(patchwork_f)
        
        datatrans4 = DataTrans(3) .set_name("datatrans4")
        datatrans4 .add_sources(projectTo) .add_destinations(plotoccupancy3)
        
        timertrans4 = TimerTrans(2) .set_name("timertrans4")
        timertrans4 .add_sources(plotoccupancy3) .add_destinations(turn2)
        
        completiontrans3 = CompletionTrans() .set_name("completiontrans3")
        completiontrans3 .add_sources(turn2) .add_destinations(patchwork_f)
        
        return self


  # now we are thinking of doing k-nearest neighbors, or k means clustering
    # want to get the hue pixels of our images (eventually, filter responses) and send into KMeans
    # to do: get the hue pixels, create the histogram

