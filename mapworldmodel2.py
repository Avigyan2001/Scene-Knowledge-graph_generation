import csv, os, math, cv2, yolo
import numpy as np
import pandas as pd
from PIL import Image  
import matplotlib.pyplot as plt 
from matplotlib import style
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import networkx as nx

 
# Output Images 
folder_img = "./rgb_images"
filename = "./pred_pose_d.txt"

target_object = "cup"
G=nx.Graph()

color_map = []
color_map.append("red")
color_map.append("blue")


classes = {'chair', 'table', 'sofa', 'tvmonitor', 'microwave', 'bed', 'oven'} #array index of classes        
# read from file

def initGraph():
    global G, classes
    for c in classes:
        G.add_node(c)
    G.add_edge("chair","table")    
    G.add_edge("sofa","table")  
    G.add_edge("sofa","chair")  
    G["chair"]["table"]['co-located'] = 0.9
    G["sofa"]["table"]['co-located'] = 0.8
    G["sofa"]["chair"]['co-located'] = 0.1

labels = []
#nx.set_edge_attributes(G, labels1, 'labels')

    
def load_images_from_folder(folder):
    images = []
    count = 1
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, 'obs_'+str(count)+'.png'))
        count += 1
        images.append(img)
    return images

def updateGraph(listParams):
    global ax1
    length = len(listParams)
    for i in range(0, length):
        if listParams[i]["class"] == target_object:
            print("TARGET Found : {} confidence: {}".format(listParams[i]["class"],listParams[i]["confidence"]))
        elif listParams[i]["class"] in classes: #check if class is known
            #updateGraph
            #
            pass

class Scene:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        

def replaceEfloat(s):
    if 'e' in s:
        sarr = s.split('e')
        return float(sarr[0]) * math.exp(float(sarr[1]))
    else:
        return float(s)

#p is used to define parallel arrays
p_arr_x = [] # x direction movements - left - right
p_arr_y = [] # y direction movements - forward - back
p_arr_x.append(0)
p_arr_y.append(0)
p_arr_angle = [0] # change in angle - rotation
p_arr_knowledge = []

with open(filename) as csv_file:
    scale_pose = 1
    csv_reader = csv.reader(csv_file, delimiter=" ")
    line_count = 0
    for row in csv_reader:
        p_arr_x.append(scale_pose * ( p_arr_x[line_count] + replaceEfloat(row[0].strip()) ) )
        p_arr_y.append(scale_pose * ( p_arr_y[line_count] + replaceEfloat(row[1].strip())) )
        p_arr_angle.append(scale_pose * replaceEfloat(row[2].strip()))
        line_count += 1

p_images = load_images_from_folder(folder_img)

style.use('ggplot')
#fig = plt.figure()

fig, (ax1, ax2) = plt.subplots(2)

G = nx.Graph()
initGraph()


plt.xlim(-5, 100)
plt.ylim(-10, 5)
graph, = plt.plot([], [], 'o')

plt.title('GeoSem Map')
plt.ylabel('Y axis')
plt.xlabel('X axis')

textvar = plt.text(100, 5, 'Image # 0', horizontalalignment='center', verticalalignment='center')

playRate = 1
counterA = 0
def animate(i):
    global counterA, textvar
    if counterA < len(p_arr_y) + 2:
        global G, ax1
        graph.set_data(p_arr_x[counterA], p_arr_y[counterA])
        plt.plot(p_arr_x[counterA:counterA+2], p_arr_y[counterA:counterA+2])
        textvar.remove()
        textvar = plt.text(100, 5, 'Image # ' + str(counterA), horizontalalignment='center', verticalalignment='center')
        [H, W, listParams] = yolo.processScene(p_images[counterA], playRate, 1) # 1 means showImage
        if listParams == []:
            arr_t = []
            arr_t.append(0)
            p_arr_knowledge.append(arr_t)
            #zone.set('No Object Detected')
        else:
            #zone.set('Zone Name')
            updateGraph(listParams)
            p_arr_knowledge.append(listParams)
            plt.scatter(p_arr_x[counterA], p_arr_y[counterA], color='black')
#            if counterA % 2 == 0:
#                G["chair"]["table"]['co-located'] = 0.5
#            else:
#                G["chair"]["table"]['co-located'] = 0.9
#            
            #plt.subplot(0).ca()
            pos = nx.spring_layout(G)
            edge_labels = nx.get_edge_attributes(G,'co-located')
            nx.draw(G, with_labels=True, ax=ax1)
            nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, ax=ax1)
            
        counterA += 1
        return graph

ani = FuncAnimation(fig, animate, frames=len(p_arr_y)-1, interval=1, repeat=False)
plt.show()
print(p_arr_knowledge)

f = open("knowledge.txt", "w")
tempStr = ""
for s in p_arr_knowledge:
    tempStr += ' '.join([str(elem) for elem in s]) + "\n"
f.write(tempStr)    
f.close()

#tag objects detected wrt knowledgebase

# determine zones wrt aggregation of detected objects
