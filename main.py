import csv, os, math, cv2, yolo
import numpy as np
import pandas as pd
from PIL import Image  
import matplotlib.pyplot as plt 
from matplotlib import style
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import networkx as nx
import time
from owlready2 import *
import types

#folder_img = "./frames"
folder_img = "./House_Room_Dataset/temp2"
onto_path = 'file://' + 'C:/Users/avigy/Desktop/SGG/ronav.owl'

#--------------------------------------------Helper Functions---------------------------------------------------------------

#-----For determining approximate colour------------
standard_colours = [[(0,0,0),'Black'],[(255,255,255),'White'],[(255,0,0),'Red'],[(0,255,0),'Lime'],[(0,0,255),'Blue'],[(255,255,0),'Yellow'],[(0,255,255),'Aqua'],[(255,0,255),'Magenta'],[(128,0,0),'Maroon'],[(0,128,0),'Green'],[(128,0,128),'Purple'],[(0,128,128),'Teal'],[(0,0,128),'Navy']]

def process_object_colour(img,params):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop_img = img[params['y']:(params['y']+params['h']), params['x']:(params['x']+params['w'])]
    rgb = np.average(crop_img, axis = (0,1)).tolist()
    min_index = 0
    index=0
    min = 10000000
    for colour in standard_colours:
      sum=0
      for k in range(3):
        sum = sum+(colour[0][k]-rgb[k])**2
      if(sum<=min):
        min=sum
        min_index = index
      index = index+1
    #print(min_index)
    obj_colour = standard_colours[min_index][1]
    return obj_colour

#--------------Creating Weight Bands-------------
dict = {}
i = 0
k = 1
while (i < 225):
  key = tuple(range(i,i+15))
  dict[key] = k
  if(i < 225/2):
    flag = 0
  else:
    flag = 1
  if((k < 8) and (flag == 0)):
    k = k + 1
  else:
    k = k - 1
  i = i + 15

def weight_band(height):
    global dict
    weight = 0
    for key in dict:
      if(height in key):
        weight = dict[key]
        break
    return weight

#--------------------Get Proximity for edges----------------
def get_proxim(G, n1, n2):
    t1 = weight_band(G.nodes[n1]['pmeters']['y'])
    t2 = weight_band(G.nodes[n1]['pmeters']['y'] + G.nodes[n1]['pmeters']['h'])
    w1 = (t1+t2)/2
    t1 = weight_band(G.nodes[n2]['pmeters']['y'])
    t2 = weight_band(G.nodes[n2]['pmeters']['y'] + G.nodes[n2]['pmeters']['h'])
    w2 = (t1+t2)/2
    mp1 = [G.nodes[n1]['pmeters']['x'] + (G.nodes[n1]['pmeters']['w'])/2, G.nodes[n1]['pmeters']['y'] +(G.nodes[n1]['pmeters']['h'])/2]
    mp2 = [G.nodes[n2]['pmeters']['x'] + (G.nodes[n2]['pmeters']['w'])/2, G.nodes[n2]['pmeters']['y'] +(G.nodes[n2]['pmeters']['h'])/2]
    wmp1 = [e * w1 for e in mp1]
    wmp2 = [e * w2 for e in mp2]

    proxim = round(math.dist(wmp1, wmp2),2)
    return proxim

#-----------Loading images------------------
def load_images_from_folder(folder):
    images = []
    count = 1
    for file in os.listdir(folder):
        #print(file)
        #img = cv2.imread(os.path.join(folder, 'din_'+str(count)+'.jpg'))
        img = cv2.imread(os.path.join(folder, file))
        count += 1
        images.append(img)
    return images
#----------------------------------------------------------------------------------------------------------------------

#--------------------------------Generating Scene Graph for Each RGB Image---------------------------------------------
t_images = load_images_from_folder(folder_img)
playRate = 1000
#listparams =[]
sg_obj_dict = {}
sg_list = []

def add_node_sg(G,img,param,dict):
    
    global sg_obj_dict
    if(param['class'] not in sg_obj_dict):
      sg_obj_dict[param['class']] = 1
    else:
      sg_obj_dict[param['class']] += 1

    if(param['class'] not in dict):
      colour = process_object_colour(img,param)
      G.add_node(param['class'] + '-'+ '1', obj_colour = colour, pmeters = param)
      dict[param['class']] = 1
    else:
      dict[param['class']] += 1
      colour = process_object_colour(img,param)
      G.add_node(param['class'] + '-' + str(dict[param['class']]), obj_colour = colour, pmeters = param)

    return G

for image in t_images:
    [H, W, listParams] = yolo.processScene(image, playRate, 1)
    G = nx.Graph()
    #print(listParams)
    object_dict={}
    for param in listParams:
        #G.add_node(param['class'])
        G = add_node_sg(G,image,param,object_dict)
    #print(G.nodes.data())
    #length = len(listParams)
    sg_visited = []
    for node1 in G.nodes:
      for node2 in G.nodes:
        if((node1 != node2) and (node1 not in sg_visited)):
          proximity = get_proxim(G, node1, node2)
          G.add_edge(node1,node2, proxim = proximity)
      sg_visited.append(node1)
    sg_list.append(G)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels = True)
    edge_labels = nx.get_edge_attributes(G,'proxim')
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
    #plt.show()
    with plt.ion():
    # interactive mode will be on
    # figures will automatically be shown
      fig1 = plt.figure()
    #plt.pause(2)
    time.sleep(3)
    plt.close('all')
    #print('------------------------------------')


#---------------------------------------Filtering and Ontology Function--------------------------------------------
l1 = []
for obj in sg_obj_dict:
    if((sg_obj_dict[obj])<1):
        l1.append(obj)
for i in l1:
    del sg_obj_dict[i]


onto = get_ontology(onto_path).load()

def add_class(keyword):
    global onto
    with onto:
        class Bedroom(Thing):
            pass
        class Living(Thing):
            pass
        class Dining(Thing):
            pass
        class Kitchen(Thing):
            pass
        NewClass = types.new_class(keyword,(Living,))

def classname_search(keyword,onto):    
    classes = list(onto.classes())
    classes_str = [x.name for x in classes]
    if(keyword not in classes_str):
        add_class(keyword)

for obj in sg_obj_dict:
    classname_search(obj,onto)

#onto.save(file = "ronav.owl", format = "rdfxml")



#-----------------------------------------Generating Knowledge Graph----------------------------------------
#onto = get_ontology(onto_path).load()

KG_living = nx.Graph()

with onto:
    class Bedroom(Thing):
        pass
    class Living(Thing):
        pass
    class Dining(Thing):
        pass
    class Kitchen(Thing):
        pass

nodes = []
temp_list = list(Living.subclasses())
#print(temp_list)
for i in temp_list:
    nodes.append(i.name)

#KG_living.add_nodes_from(nodes)
for node in nodes:
    KG_living.add_node(node, colour_list = [])

pair_count = {}

for g in sg_list:
    
    for n in g.nodes.data():
        node = n[0].split('-')[0]
        colour = n[1]['obj_colour']
        if (node in nodes):
            t_list = KG_living.nodes[node]['colour_list']
            if(colour not in t_list):
              KG_living.nodes[node]['colour_list'].append(colour)
    
    for e in g.edges:
        node1 = e[0].split('-')[0]
        node2 = e[1].split('-')[0]
        proximity = round(g.edges[e[0],e[1]]['proxim'],2)
        node_pair = (node1,node2)
        if((node_pair not in pair_count) and (node1 in nodes) and (node2 in nodes)):
            pair_count[node_pair] = 1
            KG_living.add_edge(node1, node2, proxim = proximity)
        elif((node_pair in pair_count) and (node1 in nodes) and (node2 in nodes)):
            temp_prox = KG_living.edges[node1,node2]['proxim']
            temp_prox = temp_prox * pair_count[node_pair]
            temp_prox = temp_prox + proximity
            pair_count[node_pair] +=1
            temp_prox = temp_prox/pair_count[node_pair]
            KG_living.edges[node1,node2]['proxim'] = round(temp_prox,2)

pos = nx.spring_layout(KG_living)
nx.draw(KG_living, pos, with_labels = True)
edge_labels = nx.get_edge_attributes(KG_living,'proxim')
nx.draw_networkx_edge_labels(KG_living, pos, edge_labels = edge_labels)

plt.show()

#---------------------------------------------------------------------------------------------------------------