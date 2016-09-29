import bpy
import bmesh
from mathutils import Matrix
from math import radians
from random import random
import hashlib

dir_ideal = '//ideal/'
dir_pages = '//pages/'

text_changed = True

def desel():
    for obj in bpy.context.selected_objects:
        obj.select = False

desel()

pag = bpy.data.objects['Page']
mob = bpy.data.objects['Page-morph']
txt = bpy.data.objects['Text']

mob_mkey = mob.data.shape_keys.key_blocks['Bent']
mob_mkey2 = mob.data.shape_keys.key_blocks['Bent2']
txt_md = txt.modifiers['MeshDeform']
mob_mkey.value=0
mob_mkey2.value=0

if text_changed:
    txt.select = True
    bpy.context.scene.objects.active = txt
    if txt_md.is_bound:
        bpy.ops.object.meshdeform_bind(modifier='MeshDeform')
    bpy.ops.object.meshdeform_bind(modifier='MeshDeform')
    txt.select = False
    txt.data.body="Text\nhere\n123456\n789012"

bend1_val = random()
mob_mkey.value=bend1_val
mob_mkey2.value=1-bend1_val

h = hashlib.sha256(txt.data.body.encode("utf-8"))
hs = ":".join("{:02x}".format(c) for c in h.digest()[:8])

page_file = dir_pages + hs + ".png"
bpy.data.scenes['Scene'].render.filepath = page_file
bpy.ops.render.render( write_still=True )
