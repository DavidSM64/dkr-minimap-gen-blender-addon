bl_info = {
    'name': 'DKR Minimap Generator (Experimental)',
    'blender': (2, 93, 0),
    'category': 'Object',
    'version': (0, 1, 0),
    'author': 'DavidSM64',
    'description': 'Allows you to generate a minimap image for a course model.',
}

import bpy
import math
import numpy as np
from mathutils import Vector
from bpy_extras.io_utils import ExportHelper

class MaterialVectorItem(bpy.types.PropertyGroup):
    value: bpy.props.PointerProperty(type=bpy.types.Material)

PROPS = [
    ('width', bpy.props.IntProperty(name='Width', default=32, min=1, max=128)),
    ('height', bpy.props.IntProperty(name='Height', default=32, min=1, max=128)),
    ('use_depth', bpy.props.BoolProperty(name='Indicate Depth', default=False)),
    ('depth_min', bpy.props.FloatProperty(name='Min', default=0.0, min=-1.0, max=1.0)),
    ('depth_max', bpy.props.FloatProperty(name='Max', default=1.0, min=-1.0, max=1.0)),
    ('rotation', bpy.props.IntProperty(name='Rotation (Degrees)', default=0, min=0, max=359)),
    ('scale_x', bpy.props.FloatProperty(name='Scale (Width)', default=1.0, min=0.1, max=2.0)),
    ('scale_y', bpy.props.FloatProperty(name='Scale (Height)', default=1.0, min=0.1, max=2.0)),
    ('filter_materials_bool', bpy.props.BoolProperty(name='Filter by materials', default=False)),
    ('filter_materials', bpy.props.CollectionProperty(type=MaterialVectorItem)),
    ('ignore_materials_bool', bpy.props.BoolProperty(name='Ignore materials', default=False)),
    ('ignore_materials', bpy.props.CollectionProperty(type=MaterialVectorItem)),
    ('minimap_texture', bpy.props.PointerProperty(type=bpy.types.ImageTexture))
]

###### Util ######

CHARS_TO_REMOVE=[' ', '_']
def simple_name(s):
    return ''.join(char for char in s.lower() if char not in CHARS_TO_REMOVE)

IGNORE_SIMPLE_NAMES = ['bsp', 'ignore']

def is_ignored(str):
    return any(ignoreStr in simple_name(str) for ignoreStr in IGNORE_SIMPLE_NAMES)

def print_children(obj, level=0):
    if simple_name(obj.name) in IGNORE_SIMPLE_NAMES:
        return
    print('  ' * level + obj.name)
    for child in obj.children:
        print_children(child, level + 1)

def get_min_max(obj):
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = min(bbox_corners, key=lambda corner: corner.x + corner.y + corner.z)
    max_corner = max(bbox_corners, key=lambda corner: corner.x + corner.y + corner.z)
    return min_corner, max_corner

def get_children_min_max(obj, outlist):
    if len(obj.children) == 0:
        return get_min_max(obj)
    min_corner = Vector((2**31 - 1,) * 3)
    max_corner = Vector((-2**31,) * 3)
    for child in obj.children:
        child_min, child_max = get_children_min_max(child, outlist)
        min_corner.x = min(min_corner.x, child_min.x)
        min_corner.y = min(min_corner.y, child_min.y)
        min_corner.z = min(min_corner.z, child_min.z)
        max_corner.x = max(max_corner.x, child_max.x)
        max_corner.y = max(max_corner.y, child_max.y)
        max_corner.z = max(max_corner.z, child_max.z)
        if child.type != "MESH":
            continue
        if is_ignored(child.name):
            continue
        mat_index = child.data.polygons[0].material_index
        mat = child.data.materials[mat_index]
        if bpy.context.scene.filter_materials_bool:
            doAdd = False
            for _, item in enumerate(bpy.context.scene.filter_materials):
                if mat.name == item.value.name:
                    doAdd = True
                    break
            if not doAdd:
                continue
        if bpy.context.scene.ignore_materials_bool:
            doAdd = True
            for _, item in enumerate(bpy.context.scene.ignore_materials):
                if mat.name == item.value.name:
                    doAdd = False
                    break
            if not doAdd:
                continue
        outlist.append(child)
    return min_corner, max_corner

longest_image_axis = 0
shorter_image_axis = 0

def create_camera(context, orthScale):
    # Create a new camera
    camera_data = bpy.data.cameras.new(name='MinimapCamera')
    camera_data.type = 'ORTHO'

    if longest_image_axis == 0:
        raise SystemError("longest_image_axis is zero!")

    #totalShiftAmount = (((1 - (60/longest_image_axis)) * orthScale) / 2) * 0.1
    #angle = math.radians(context.scene.rotation + 90)
    #camera_data.shift_x = math.cos(angle) * totalShiftAmount
    #camera_data.shift_y = math.sin(angle) * totalShiftAmount
            
    orthScale *= (longest_image_axis / 60)
    camera_data.ortho_scale = orthScale

    # Set camera resolution
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256

    aspectScale = context.scene.scale_x / context.scene.scale_y

    # Set camera aspect ratio
    bpy.context.scene.render.pixel_aspect_x = 1
    bpy.context.scene.render.pixel_aspect_y = 1
    
    if aspectScale > 1:
        bpy.context.scene.render.pixel_aspect_y = aspectScale
    elif aspectScale < 1:
        bpy.context.scene.render.pixel_aspect_x = 1 / aspectScale

    camera_object = bpy.data.objects.new('MinimapCamera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    return camera_data, camera_object

def render_objects_save_current_state():
    saveState = {}
    saveState["prevResolutionX"] = bpy.context.scene.render.resolution_x
    saveState["prevResolutionY"] = bpy.context.scene.render.resolution_y
    saveState["prevAspectX"] = bpy.context.scene.render.pixel_aspect_x
    saveState["prevAspectY"] = bpy.context.scene.render.pixel_aspect_y
    saveState["prevCamera"] = bpy.context.scene.camera
    saveState["useNodes"] = bpy.context.scene.use_nodes
    saveState["useCompositing"] = bpy.context.scene.render.use_compositing
    saveState["usePassZ"] = bpy.context.scene.view_layers[0].use_pass_z
    saveState["objectsHideRender"] = []
    for obj in bpy.context.scene.objects:
        saveState["objectsHideRender"].append(obj.hide_render)
    return saveState

def render_objects_load_current_state(saveState):
    bpy.context.scene.render.resolution_x = saveState["prevResolutionX"]
    bpy.context.scene.render.resolution_y = saveState["prevResolutionY"]
    bpy.context.scene.render.pixel_aspect_x = saveState["prevAspectX"]
    bpy.context.scene.render.pixel_aspect_y = saveState["prevAspectY"]
    bpy.context.scene.use_nodes = saveState["useNodes"]
    bpy.context.scene.render.use_compositing = saveState["useCompositing"]
    bpy.context.scene.view_layers[0].use_pass_z = saveState["usePassZ"]
    bpy.context.scene.camera = saveState["prevCamera"]
    for i, obj in enumerate(bpy.context.scene.objects):
        try:
            obj.hide_render = saveState["objectsHideRender"][i]
        except IndexError:
            break

def crop_image_in_place(orig_img, cropped_min_x, cropped_max_x, cropped_min_y, cropped_max_y):
    num_channels = orig_img.channels
    cropped_size_x = cropped_max_x - cropped_min_x
    cropped_size_y = cropped_max_y - cropped_min_y
    orig_size_x = orig_img.size[0]
    orig_size_y = orig_img.size[1]
    current_cropped_row = 0
    new_pixels = []
    for yy in range(orig_size_y - cropped_max_y, orig_size_y - cropped_min_y):
        orig_start_index = (cropped_min_x + yy*orig_size_x) * num_channels
        orig_end_index = orig_start_index + (cropped_size_x * num_channels)
        new_pixels.extend(orig_img.pixels[orig_start_index:orig_end_index])
        current_cropped_row += 1
    orig_img.scale(cropped_size_x, cropped_size_y)
    orig_img.pixels = new_pixels

def extend_canvas(image: bpy.types.Image, top_margin: int, bottom_margin: int, left_margin: int, right_margin: int):
    # print("adding margins:", (top_margin, bottom_margin, left_margin, right_margin))
    # Get the width and height of the original image
    width, height = image.size

    # Calculate the size of the extended canvas
    new_width = width + left_margin + right_margin
    new_height = height + top_margin + bottom_margin

    # Convert the list of RGBA float values into a numpy array
    image_np = np.array(image.pixels).reshape((height, width, 4))

    # Create a new numpy array with the extended canvas
    new_image_np = np.zeros((new_height, new_width, 4), dtype=image_np.dtype)
    new_image_np[top_margin:top_margin+height, left_margin:left_margin+width] = image_np

    # Update the size and pixels attributes of the input bpy.types.Image object
    image.scale(new_width, new_height)
    image.pixels = new_image_np.flatten()

curTex = None
longestAxis = 'X'

def render_objects_with_depth(context, objects, camera_location, camera_target, orthScale):
    global curTex
    saveState = render_objects_save_current_state()
    try:
        try:
            if curTex != None:
                bpy.data.images.remove(curTex.image)
                bpy.data.textures.remove(curTex)
                curTex = None
        except:
            pass

        camera_data, camera_object = create_camera(context, orthScale)

        # Set the camera position
        camera_object.location = camera_location

        # Set the camera target
        direction = Vector(camera_target) - camera_object.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera_object.rotation_euler = rot_quat.to_euler()

        # Set the camera as the active camera for the scene
        bpy.context.scene.camera = camera_object

        # Hide all objects in the scene
        for obj in bpy.context.scene.objects:
            obj.hide_render = True

        # Unhide the objects to be rendered
        for obj in objects:
            obj.hide_render = False

        bpy.context.scene.use_nodes = True

        # Useful scene variables
        scn = bpy.context.scene
        cam = camera_object
        output_path = bpy.utils.resource_path('USER') + "/image.png"
        tree = bpy.context.scene.node_tree
        links = tree.links

        bpy.context.scene.render.use_compositing = True
        bpy.context.scene.view_layers[0].use_pass_z = True

        for n in tree.nodes:
            tree.nodes.remove(n)
        rl = tree.nodes.new('CompositorNodeRLayers')
        vl = tree.nodes.new('CompositorNodeViewer')
        vl.use_alpha = True
        links.new(rl.outputs[0], vl.inputs[0])  # link Image to Viewer Image RGB
        links.new(rl.outputs['Depth'], vl.inputs[1])  # link Render Z to Viewer Image Alpha

        bpy.context.scene.camera.rotation_euler[2] += math.radians(context.scene.rotation)

        # Render the scene and save the image to a file
        #bpy.context.scene.render.filepath = bpy.utils.resource_path('USER') + "/image.png"
        bpy.ops.render.render()

        #bpy.data.images['Viewer Node'].save_render(filepath=bpy.utils.resource_path('USER') + "/image.png")

        w = 256
        h = 256
        pixels = np.array(bpy.data.images['Viewer Node'].pixels)
        image_with_depth = pixels.reshape(h,w,4)
        z_values = sorted(np.unique(image_with_depth[:, :, 3]))
        z_min = z_values[0]
        try:
            z_max = z_values[-2]
        except:
            z_max = z_values[-1]
        z_nohit = 65000
        outPixels = []
        unique = []
        for y in range(h):
            for x in range(w):
                depth = image_with_depth[y, x, 3]
                if depth > z_nohit:
                    continue
                if not depth in unique:
                    unique.append(depth)
        unique.sort(reverse=True)
        #print(len(unique), unique)
        uniqueLen = len(unique)
        for y in range(h):
            for x in range(w):
                depth = image_with_depth[y, x, 3]
                if depth >= z_nohit:
                    outPixels += [1.0, 1.0, 1.0, 0.0]
                    continue
                val = 1.0
                if context.scene.use_depth:
                    val = (unique.index(depth) / uniqueLen)
                    if val < context.scene.depth_min:
                        val = 0.0
                    elif val > context.scene.depth_max:
                        val = 1.0
                    else:
                        val = 1.0 - ((context.scene.depth_max - val) / (context.scene.depth_max - context.scene.depth_min))
                    val = 0.6 + (val * 0.4)
                outPixels += [val, val, val, 1.0]

        image = bpy.data.images.new("minimapImage", width=w, height=h)
        image.colorspace_settings.name = 'Filmic sRGB'
        image.pixels = outPixels

        w = 60 # context.scene.width
        h = 60 # context.scene.height
        
        image.scale(longest_image_axis, longest_image_axis)
        cosAngle = math.cos(math.radians(context.scene.rotation + 90))
        sinAngle = math.sin(math.radians(context.scene.rotation + 90))
        if longestAxis == 'X':
            a0 = (longest_image_axis - shorter_image_axis) // 2
            a1 = a0 + shorter_image_axis
            b0 = 0
            b1 = b0 + longest_image_axis
        else:
            a0 = 0
            a1 = a0 + longest_image_axis
            b0 = (longest_image_axis - shorter_image_axis) // 2
            b1 = b0 + shorter_image_axis
        x0 = (cosAngle * a0) + (sinAngle * b0)
        x1 = (cosAngle * a1) + (sinAngle * b1)
        y0 = (sinAngle * a0) + (cosAngle * b0)
        y1 = (sinAngle * a1) + (cosAngle * b1)
        x0 = int(x0)
        x1 = int(x1)
        y0 = int(y0)
        y1 = int(y1)
        if x0 < y0:
            y0 -= 1
            y1 -= 1
        else:
            x0 -= 1
            x1 -= 1
        
        #print((a0, a1, b0, b1), (x0, x1, y0, y1))

        if shorter_image_axis < longest_image_axis:
            crop_image_in_place(image, x0, x1, y0, y1)
        
        if x1 < context.scene.width or y1 < context.scene.height:
            extend_canvas(image, 0, context.scene.height - image.size[1], 0, context.scene.width - image.size[0])

        imgW, imgH = image.size

        if imgW > context.scene.width or imgH > context.scene.height:
            x0 = abs(imgW - context.scene.width) // 2
            x1 = x0 + context.scene.width
            y0 = abs(imgH - context.scene.height) // 2
            y1 = y0 + context.scene.height
            x0 = int(x0)
            x1 = int(x1)
            y0 = int(y0)
            y1 = int(y1)
            crop_image_in_place(image, x0, x1, y0, y1)

        texture = bpy.data.textures.new(name="minimap", type="IMAGE")
        texture.image = image
        curTex = bpy.data.textures['minimap']
        curTex.extension = 'CLIP'

        context.scene.minimap_texture = curTex

        bpy.data.objects.remove(camera_object)
        bpy.data.cameras.remove(camera_data)

        # Load back the previous state
        render_objects_load_current_state(saveState)
    except:
        # Load back the previous state
        render_objects_load_current_state(saveState)
        raise


def get_center(min_point: tuple, max_point: tuple) -> tuple:
    x_center = (min_point[0] + max_point[0]) / 2
    y_center = (min_point[1] + max_point[1]) / 2
    z_center = (min_point[2] + max_point[2]) / 2
    return (x_center, y_center, z_center)


###### Operators ######

class MinimapGenOperator(bpy.types.Operator):
    
    bl_idname = 'opr.object_renamer_operator'
    bl_label = 'Object Renamer'
    
    def execute(self, context):
        global longest_image_axis, shorter_image_axis, longestAxis
        for obj in bpy.context.scene.objects:
            obj.hide_render = False
            if obj.parent is None and obj.type == "EMPTY" and not is_ignored(obj.name):
                modelObj = obj

        outlist = []
        min_corner, max_corner = get_children_min_max(modelObj, outlist)
        #print(min_corner, max_corner)
        if len(outlist) == 0:
            print('No objects to render!')
        else:
            center = get_center(min_corner, max_corner)
            center_above = (center[0], center[1], max_corner[2] + 10)
            scaleX = max_corner[0] - min_corner[0]
            scaleZ = max_corner[1] - min_corner[1]
            #print('aspect:', (scaleX / scaleZ), (scaleZ / scaleX))
            orthScale = scaleZ # X scale doesn't actually matter?
            if context.scene.scale_x < context.scene.scale_y:
                orthScale *= 1 / context.scene.scale_x
            elif context.scene.scale_y < context.scene.scale_x:
                orthScale *= 1 / context.scene.scale_y
            elif context.scene.scale_x != 1.0:
                orthScale *= 1 / context.scene.scale_x

            imgAxisX = 60.0 * (context.scene.scale_x) * (scaleX / scaleZ)
            imgAxisZ = 60.0 * (context.scene.scale_y)

            w = 60 # context.scene.width
            h = 60 # context.scene.height

            angle = math.radians(context.scene.rotation + 90)

            if scaleX > scaleZ:
                #longest_image_axis = max(math.ceil((60.0 * context.scene.scale_x) * (scaleX / scaleZ)), longest_image_axis)
                #longest_image_axis = (math.cos(angle) * imgAxisX * (60/w)) + (math.sin(angle) * imgAxisZ * (60/h))
                longest_image_axis = imgAxisX * (60.0 / w)
                shorter_image_axis = longest_image_axis * (scaleZ / scaleX)
                longestAxis = 'X'
            else:
                #longest_image_axis = max(math.ceil((60.0 * context.scene.scale_y)), longest_image_axis)
                #longest_image_axis = (math.sin(angle) * imgAxisX * (60/w)) + (math.cos(angle) * imgAxisZ * (60/h))
                longest_image_axis = imgAxisZ * (60.0 / h)
                shorter_image_axis = longest_image_axis * (scaleX / scaleZ)
                longestAxis = 'Z'

            longest_image_axis = round(longest_image_axis)
            shorter_image_axis = round(shorter_image_axis)
            #print(longestAxis, longest_image_axis)

            render_objects_with_depth(context, outlist, center_above, center, orthScale)
            
        return {'FINISHED'}

class FilterMaterialsAddItem(bpy.types.Operator):
    bl_idname = "filter_materials.add_item"
    bl_label = "Add Item"

    def execute(self, context):
        obj = context.scene
        item = obj.filter_materials.add()
        item.value = None
        return {'FINISHED'}

class FilterMaterialsRemoveItem(bpy.types.Operator):
    bl_idname = "filter_materials.remove_item"
    bl_label = "Remove Item"

    def execute(self, context):
        obj = context.scene
        obj.filter_materials.remove(len(obj.filter_materials) - 1)
        return {'FINISHED'}

class IgnoreMaterialsAddItem(bpy.types.Operator):
    bl_idname = "ignore_materials.add_item"
    bl_label = "Add Item"

    def execute(self, context):
        obj = context.scene
        item = obj.ignore_materials.add()
        item.value = None
        return {'FINISHED'}

class IgnoreMaterialsRemoveItem(bpy.types.Operator):
    bl_idname = "ignore_materials.remove_item"
    bl_label = "Remove Item"

    def execute(self, context):
        obj = context.scene
        obj.ignore_materials.remove(len(obj.ignore_materials) - 1)
        return {'FINISHED'}
    
class SaveMinimapAsOperator(bpy.types.Operator, ExportHelper):
    bl_idname = "image.save_minimap_as"
    bl_label = "Save As"
    filename_ext = ".png"

    def execute(self, context):
        # get the image
        img = bpy.data.images['minimapImage']
        
        # save the image to the selected file path
        img.save_render(filepath=self.filepath)
        return {'FINISHED'}

###### Panel ######

class MinimapGenPanel(bpy.types.Panel):
    
    bl_idname = 'VIEW3D_PT_object_renamer'
    bl_label = 'Minimap Generator'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'DKR'
    
    def draw(self, context):
        col = self.layout.column()
        
        col.prop(context.scene, "width")
        col.prop(context.scene, "height")
        col.prop(context.scene, "scale_x")
        col.prop(context.scene, "scale_y")
        col.prop(context.scene, "rotation")
        col.prop(context.scene, "use_depth")

        if context.scene.use_depth:
            col2 = col.row(align=True)
            col2.prop(context.scene, "depth_min")
            col2.prop(context.scene, "depth_max")

        col.prop(context.scene, "filter_materials_bool")

        if context.scene.filter_materials_bool:
            col2 = col.row(align=True)
            col2.label(text="Material to use:")
            col2.operator("filter_materials.add_item", icon='ADD', text="")
            col2.operator("filter_materials.remove_item", icon='REMOVE', text="")
            for i, item in enumerate(context.scene.filter_materials):
                col.prop(item, "value", text="")

        col.prop(context.scene, "ignore_materials_bool")
        if context.scene.ignore_materials_bool:
            col2 = col.row(align=True)
            col2.label(text="Material to ignore:")
            col2.operator("ignore_materials.add_item", icon='ADD', text="")
            col2.operator("ignore_materials.remove_item", icon='REMOVE', text="")
            for i, item in enumerate(context.scene.ignore_materials):
                col.prop(item, "value", text="")

        col.operator('opr.object_renamer_operator', text='Generate Minimap')
        if curTex != None:
            col.template_ID_preview(context.scene, "minimap_texture", new="texture.new", hide_buttons=True)
            col.operator('image.save_minimap_as', text='Save minimap as...')

###### Main ######

CLASSES = [
    MinimapGenOperator,
    MinimapGenPanel,
    FilterMaterialsAddItem,
    FilterMaterialsRemoveItem,
    IgnoreMaterialsAddItem,
    IgnoreMaterialsRemoveItem,
    SaveMinimapAsOperator
]

def register():
    bpy.utils.register_class(MaterialVectorItem)

    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)
    
    for klass in CLASSES:
        bpy.utils.register_class(klass)

def unregister():
    bpy.utils.unregister_class(MaterialVectorItem)

    for (prop_name, _) in PROPS:
        delattr(bpy.types.Scene, prop_name)

    for klass in CLASSES:
        bpy.utils.unregister_class(klass)
        

if __name__ == '__main__':
    register()