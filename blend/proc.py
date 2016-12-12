import bpy
import bmesh
from mathutils import Matrix
from math import radians
from random import random
import hashlib
import os.path

dir_ideal = '//ideal'
dir_pages = '//pages'

morph_versions = 100  # Number of warped page images to create
max_wordsets = 900

def desel():
    for obj in bpy.context.selected_objects:
        obj.select = False

def text_rebind():
    global txt
    morph_clear()
    txt.select = True
    bpy.context.scene.objects.active = txt
    if txt_md.is_bound:
        bpy.ops.object.meshdeform_bind(modifier='MeshDeform')
    bpy.ops.object.meshdeform_bind(modifier='MeshDeform')
    txt.select = False

def init():
    global pag, mob, txt, mob_mkey, mob_mkey2, txt_md
    pag = bpy.data.objects['Page']
    mob = bpy.data.objects['Page-morph']
    txt = bpy.data.objects['Text']
    mob_mkey = mob.data.shape_keys.key_blocks['Bent']
    mob_mkey2 = mob.data.shape_keys.key_blocks['Bent2']
    txt_md = txt.modifiers['MeshDeform']
def morph_clear():
    global mob_mkey, mob_mkey2
    mob_mkey.value=0
    mob_mkey2.value=0    
def morph_rand():
    global bend1_val, mob_mkey, mob_mkey2
    bend1_val = random()
    mob_mkey.value=bend1_val
    mob_mkey2.value=1-bend1_val

init()
desel()

file = open("words.txt", "r")
wordsetcount = 0
for line in file:
    wordsetcount = wordsetcount+1
    if wordsetcount > max_wordsets:
        break
    line = line.strip("\n\r").replace("\\n","\n")
    txt.data.body = line
    # morph_clear() # done by text_rebind()
    text_rebind()

    h = hashlib.sha256(txt.data.body.encode("utf-8"))
    hs = "".join("{:02x}".format(c) for c in h.digest()[:8])

    ideal_filepath = "{}/{}".format(dir_ideal, hs)
    ideal_filepath = bpy.path.abspath(ideal_filepath)
    if os.path.isdir(ideal_filepath) == False:
        os.mkdir(ideal_filepath)
    ideal_file = "{}/0001.png".format(ideal_filepath)
    print("Ideal file: " + ideal_file)
    if os.path.isfile(ideal_file) == False:
        print("Rendering ideal output at " + ideal_file)
        bpy.data.scenes['Scene'].render.filepath = ideal_file
        bpy.ops.render.render( write_still=True )
    
    page_filepath = "{}/{}".format(dir_pages, hs)
    page_filepath = bpy.path.abspath(page_filepath)
    
    if os.path.isdir(page_filepath) == False:
        os.mkdir(page_filepath)
    for i in range(0, morph_versions):
        morph_rand()
        inc = 0
        while True:
            page_file = "{}/{:04d}.png".format(page_filepath, inc)
            if os.path.isfile(page_file) == False:
                print("Available filename found at: " + page_file)
                break
            inc = inc+1
        if inc < morph_versions:
            bpy.data.scenes['Scene'].render.filepath = page_file
            bpy.ops.render.render( write_still=True )
        