import os
from owlready2 import *
import types

onto_path = 'file://' + 'C:/Users/avigy/Desktop/SGG/ronav.owl'
onto = get_ontology(onto_path).load()

#classes = list(onto.classes())
#print('All classes\n'+'-'*20)
'''for x in classes: 
    #print(x)
    if(x.name == 'chair'):
        cls = x
#print()'''

#cls = 'ronav.bed'
#print(cls)
#print(cls.__dict__)

'''properties = list(onto.object_properties())
#print('\nAll object properties\n'+'-'*20)
for x in properties: 
    print(x)
    if(x.name == 'inFrontOf'):
        prp = x
print()'''

#print(prp.__dict__)
#print(onTopof.domain)

with onto:   
    class Living(Thing):
        pass
    '''class chair(Thing):
        pass
    class onTopOf(ObjectProperty):
        #domain = [clock]
        #range  = [chair]
        pass
    class inFrontOf(ObjectProperty):
        pass
    class IndoorObject(Thing):
        pass
    NewClass = types.new_class("clock", (IndoorObject,))'''

print(list(Living.subclasses()))
list1 = list(Living.subclasses())
for i in list1:
    print(i.name)
#print(inFrontOf.domain[0].name)
#print(IndoorObject.descendants())

def add_class(keyword,onto):
    with onto:
        class IndoorObject(Thing):
            pass
        NewClass = types.new_class(keyword,(IndoorObject,))

def classname_search(keyword,onto):    
    classes = list(onto.classes())
    classes_str = [x.name for x in classes]
    if(keyword not in classes_str):
        add_class(keyword,onto)

#str = "clock"
#classname_search(str,onto)
#onto.save(file = "ronav.owl", format = "rdfxml")