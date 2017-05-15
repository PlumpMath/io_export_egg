# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8-80 compliant>

import bpy

import os.path
import time
import codecs
import shutil

import numpy.linalg as la
import numpy as np
import math
from math import radians
from math import pi
from mathutils import *

import bmesh

from bpy_extras.io_utils import ExportHelper
from bpy.props import (
        StringProperty,
        BoolProperty,
        EnumProperty,
        )

bl_info = {
    "name": "Export Panda3D Format (.egg)",
    "description": "Export image textures, materials, meshes and bones "
                   "from Makehuman model to Panda3D egg",
    "author": "Milan Vontorcik",
    "version": (1, 0),
    "blender": (2, 70, 0),
    "location": "File > Export > Panda3D Format (.egg)",
    "warning": "",
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.7/Py/"
                "Scripts/Import-Export/Panda3D_Exporter",
    "category": "Import-Export",
}

_Identity = np.identity(4, float)


def good_file_name(filepath):
    filename = os.path.basename(filepath)
    name = os.path.splitext(filename)[0]
    string = name.replace(' ', '_').replace('-', '_')
    return string


def good_bone_name(bone_name):
    return bone_name.replace('.', '_').replace('-', '_')


def good_mesh_name(mesh_name):
    return mesh_name.split(':')[-1]


def good_texture_name(filepath):
    texfile = os.path.basename(filepath)
    return texfile.replace('.', '_')


def good_material_name(material_name):
    return material_name.replace(' ', '_').split(':')[-1]


class ExportEggWorker(object):
    def __init__(self, human, weight_precision, filepath, use_rel_paths):
        self._use_rel_paths = True
        self._human = human
        self._vertex_weight_precision = '{0:.' + str(weight_precision) + 'f}'
        self._copied_files = {}
        self._egg_fp = None
        self._tex_folder = None
        self._outFolder = ''
        self._separate_tex_folder = 'textures'
        self._filepath = filepath
        self._filename = good_file_name(filepath)
        self._setup_tex_folder()
        return None

    def _get_sub_folder(self):
        folder = os.path.join(self._outFolder, self._separate_tex_folder)
        if not os.path.exists(folder):
            try:
                os.mkdir(folder)
            except:
                print('Unable to create separate folder:' % folder)
                return None
        return folder

    def _setup_tex_folder(self):

        self._outFolder = os.path.realpath(os.path.dirname(self._filepath))
        self._tex_folder = self._get_sub_folder()

    def _copy_texture_to_new_location(self, image):
        filepath = image.filepath
        filename = os.path.basename(filepath)

        newpath = os.path.abspath(os.path.join(self._tex_folder, filename))
        try:
            self._copied_files[filepath]
            done = True
        except:
            done = False
        if not done:
            try:
                image.save_render(newpath)
            except:
                print('Unable to copy \'%s\' -> \'%s\'' % (filepath, newpath))
            self._copied_files[filepath] = True

        if self._use_rel_paths:
            relpath = os.path.relpath(newpath, self._outFolder)
            return str(os.path.normpath(relpath))
        else:
            return newpath

    def _write_textures(self, rmeshes):
        for rmesh in rmeshes:
            materials = rmesh.data.materials
            for mat in materials:
                if mat:
                    imageTextures = [(tslot.texture,
                                      tslot.texture_coords, tslot.mapping)
                                     for tslot in mat.texture_slots.values()
                                     if tslot is not None and
                                     tslot.texture.type == 'IMAGE' and
                                     getattr(tslot.texture.image, 'source',
                                             '') == 'FILE']
                    for (texture, texture_coords, mapping) in imageTextures:
                        if texture.image.filepath not in self._copied_files:
                            self._write_texture(texture, texture_coords,
                                                mapping)

    def _get_image_format(self, image):
        colorspaceName = image.colorspace_settings.name
        retVal = ''
        if colorspaceName == 'sRGB':
            retVal = 'RGB'
        else:
            retVal = ''
            print ('undefined colorspace_settings.name %s for texture image %s'
                   % (colorspace_settings.name, image.filepath))
        if image.use_alpha:
            retVal += 'A'
        return retVal

    def _get_texture_wrap_u(self, texture):
        if texture.extension == 'REPEAT' and texture.repeat_x == 1:
            return 'repeat'
        else:
            print ('undefined extension %s ' % texture.extension +
                   'for texture image %s, ' % texture.name +
                   'texture.repeat_x=%f' % texture.repeat_x)

    def _get_texture_wrap_v(self, texture):
        if texture.extension == 'REPEAT' and texture.repeat_y == 1:
            return 'repeat'
        else:
            print ('undefined extension %s ' % texture.extension +
                   'for texture image %s, ' % texture.name +
                   'texture.repeat_y=%f' % texture.repeat_y)

    def _get_texture_type(self, texture_coords, mapping):
        if texture_coords == 'UV' and mapping == 'FLAT':
            return '2d'
        else:
            print ('undefined texture type for texture_coords %s '
                   % texture_coords +
                   'and mapping=%s' % mapping)
            return 'undefined'

    def _write_texture(self, texture, texture_coords, mapping):
        if not texture:
            return
        image = texture.image
        if not image:
            return
        newpath = self._copy_texture_to_new_location(image)
        texname = good_texture_name(image.filepath)
        self._egg_fp.write(
            '<Texture> %s {\n' % texname +
            '  "%s"\n' % newpath +
            '  <Scalar> format { %s }\n' % self._get_image_format(image) +
            '  <Scalar> wrapu { %s }\n' % self._get_texture_wrap_u(texture) +
            '  <Scalar> wrapv { %s }\n' % self._get_texture_wrap_v(texture) +
            '  <Scalar> type { %s }\n' % self._get_texture_type(texture_coords,
                                                                mapping) +
            '}\n\n'
        )

    def _write_materials(self, rmeshes):
        for rmesh in rmeshes:
            mat = rmesh.active_material
            material_name = good_material_name(mat.name)
            self._egg_fp.write(
                '<Material> %s {\n' % material_name +
                '  <Scalar> diffr { %.4f }\n' % mat.diffuse_color.r +
                '  <Scalar> diffg { %.4f }\n' % mat.diffuse_color.g +
                '  <Scalar> diffb { %.4f }\n' % mat.diffuse_color.b +
                '  <Scalar> specr { %.4f }\n' % mat.specular_color.r +
                '  <Scalar> specg { %.4f }\n' % mat.specular_color.g +
                '  <Scalar> specb { %.4f }\n' % mat.specular_color.b +
                '  <Scalar> speca { %.4f }\n' % mat.specular_alpha +
                '  <Scalar> ambr { %.4f }\n' % mat.ambient +
                '  <Scalar> ambg { %.4f }\n' % mat.ambient +
                '  <Scalar> ambb { %.4f }\n' % mat.ambient +
                '  <Scalar> emitr { %.4f }\n' % mat.emit +
                '  <Scalar> emitg { %.4f }\n' % mat.emit +
                '  <Scalar> emitb { %.4f }\n' % mat.emit +
                '  <Scalar> shininess { %.4f }\n' % mat.specular_hardness +
                '}\n\n')

    def _mesh_to_weight_dict(self, mesh_obj):
        """
        Takes a mesh and return its group names and a dict of dicts,
        one dict per vertex.
        aligning the each vert dict with the group names,
        each dict contains float value for the weight.
        """
        group_names = [g.name for g in mesh_obj.vertex_groups]
        group_names_tot = len(group_names)
        mesh = mesh_obj.data
        weight_dict = {}
        if not group_names_tot:
            # no verts? return empty dictionary
            return group_names, weight_dict
        for v in mesh.vertices:
            for g in v.groups:
                # possible weights are out of range
                g_index = g.group
                if g_index < group_names_tot:
                    if v.index in weight_dict:
                        weight_dict[v.index][g_index] = g.weight
                    else:
                        weight_dict[v.index] = {}
                        weight_dict[v.index][g_index] = g.weight
        return group_names, weight_dict

    def _meshes_to_weight_dict(self):
        mesh_names = [child.name
                      for child in self._human.children
                      if child.type == 'MESH']
        bone_weight_dict = {bone.name: {} for bone in self._human.data.bones}
        for bone in self._human.data.bones:
            for mesh_name in mesh_names:
                bone_weight_dict[bone.name][mesh_name] = {}
        for mesh_name in mesh_names:
            mesh = bpy.data.objects[mesh_name]
            group_names, weight_dict = self._mesh_to_weight_dict(mesh)
            for v_index in weight_dict:
                v_groups = weight_dict[v_index]
                for g_index in v_groups:
                    weight = v_groups[g_index]
                    bone_name = group_names[g_index]
                    weight_str = self._vertex_weight_precision.format(weight)
                    weights = bone_weight_dict[bone_name][mesh_name]
                    if weight != 0.0:
                        if weight_str not in weights:
                            weights[weight_str] = [v_index]
                        else:
                            weights[weight_str].append(v_index)
        return bone_weight_dict

    def _write_armature(self, meshes, name, indent_level=0):
        padding = indent_level*' '
        skel = self._human.data
        bone_weight_dict = self._meshes_to_weight_dict()
        roots = [bone
                 for bone in skel.bones
                 if bone.parent is None]
        self._write_bone(roots[0], bone_weight_dict, indent_level)

    def _write_bone_vertex_ref(self, meshes_weights, indent_level):
        padding = indent_level*' '
        for mesh_name in sorted(meshes_weights):
            weights_dict = meshes_weights[mesh_name]
            meshName = good_mesh_name(mesh_name)
            for weight in sorted(weights_dict):
                vertex_indices = weights_dict[weight]
                self._egg_fp.write('%s<VertexRef> {\n' % padding +
                                   '%s  ' % padding)
                for vertex_index in sorted(vertex_indices):
                    self._egg_fp.write('%d ' % vertex_index)
                self._egg_fp.write(
                    '\n%s  <Scalar> membership { %s }\n' % (padding, weight) +
                    '%s  <Ref> { %s_Mesh }\n' % (padding, meshName) +
                    '%s}\n' % padding)

    def _write_bone(self, bone, bones_weights, indent_level=0):
        bname = good_bone_name(bone.name)
        padding = indent_level*' '
        self._egg_fp.write('%s<Joint> %s {\n' % (padding, bname) +
                           '%s  <Transform> {\n' % padding
                           )
        self._write_bone_translation(bone, indent_level+4)
        self._egg_fp.write('%s  }\n' % padding)
        for child_bone in bone.children:
            self._write_bone(child_bone, bones_weights, indent_level+2)
        self._write_bone_vertex_ref(bones_weights[bone.name], indent_level+2)
        self._egg_fp.write('%s}\n' % padding)

    def _write_vertex(self, vertex, indent_level, bvertex, uv_layer, mesh,
                      group_names, weights):
        padding = indent_level*' '
        uv_first = self._uv_from_vert_first(uv_layer, bvertex)
        uv_set = self._uv_from_vert_set(uv_layer, bvertex, mesh)
        self._egg_fp.write('%s<Vertex> %d {\n' % (padding, vertex.index) +
                           '%s  %.4f ' % (padding, vertex.co[0]) +
                           '%.4f %.4f\n' % (vertex.co[1], vertex.co[2])
                           )
        self._egg_fp.write('%s  <UV> { ' % padding +
                           '%.4f %.4f }\n' % (uv_first[0], uv_first[1]))
        self._egg_fp.write('%s  <Normal> { ' % padding +
                           '%.4f ' % vertex.normal[0] +
                           '%.4f ' % vertex.normal[1] +
                           '%.4f }\n' % vertex.normal[2])
        for index, weight in weights.items():
            bname = good_bone_name(group_names[index])
            if weight != 0.0:
                self._egg_fp.write('%s  // %s:' % (padding, bname) +
                                   '%.8f\n' % weight)
        self._egg_fp.write('%s}\n' % padding)

    def _uv_from_vert_first(self, uv_layer, v):
        for link_loop in v.link_loops:
            uv_data = link_loop[uv_layer]
            return uv_data.uv
        return None

    def _uv_from_vert_set(self, uv_layer, bvertex, mesh):
        uv_set = []
        for loop in bvertex.link_loops:
            if loop[uv_layer].uv not in uv_set:
                uv_set.append(loop[uv_layer].uv)
        mat = mesh.materials[loop.face.material_index]
        material_name = good_material_name(mat.name)
        texture_names = [good_texture_name(texture_slot.texture.image.filepath)
                         for texture_slot in mat.texture_slots.values()
                         if texture_slot is not None and
                         texture_slot.texture.type == 'IMAGE']
        tex_uv = []
        if len(texture_names) == 1 and len(uv_set) == 1:
            tex_uv = [(texture_names[0], uv_set[0])]
        elif len(texture_names) == 1:
            tex_uv = [(texture_names[0], uv) for uv in uv_set]
        elif len(uv_set) == 1:
            tex_uv = [(tex, uv_set[0]) for tex in texture_names]
        return tex_uv

    def _write_mesh_object(self, mesh_obj, indent_level=0):
        padding = indent_level*' '
        mesh_name = good_mesh_name(mesh_obj.name)
        self._egg_fp.write('%s  <Group> %s_Mesh {\n' % (padding, mesh_name))
        self._write_vertexPool(mesh_obj, indent_level+4)
        self._write_polygons(mesh_obj.data, indent_level+4)
        self._egg_fp.write('%s  }\n' % padding)

    def _write_vertexPool(self, mesh_obj, indent_level=0):
        padding = indent_level*' '
        mesh = mesh_obj.data
        mesh_name = good_mesh_name(mesh.name)
        self._egg_fp.write('%s<VertexPool> %s_Mesh {\n' % (padding, mesh_name))
        group_names, weight_dict = self._mesh_to_weight_dict(mesh_obj)
        bm = bmesh.new()   # create an empty BMesh
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv.active
        for vertex in mesh.vertices:
            self._write_vertex(vertex, indent_level+2, bm.verts[vertex.index],
                               uv_layer, mesh, group_names,
                               weight_dict[vertex.index])
        bm.free()
        self._egg_fp.write('%s}\n' % padding)

    def _write_polygons(self, mesh, indent_level=0):
        padding = indent_level*' '
        mesh_name = good_mesh_name(mesh.name)
        for poly in mesh.polygons:
            mat = mesh.materials[poly.material_index]
            material_name = good_material_name(mat.name)
            texture_names = [good_texture_name(tslot.texture.image.filepath)
                             for tslot in mat.texture_slots.values()
                             if tslot is not None and
                             tslot.texture.type == 'IMAGE']
            self._egg_fp.write('%s<Polygon> %d {\n' % (padding, poly.index))
            for tex_name in texture_names:
                self._egg_fp.write('%s  <TRef> { %s }\n' % (padding, tex_name))
            self._egg_fp.write('%s  <MRef> { ' % padding +
                               '%s }\n' % material_name +
                               '%s  <VertexRef> { \n' % padding +
                               '%s    ' % padding
                               )
            for loop_index in poly.loop_indices:
                self._egg_fp.write('%d ' % mesh.loops[loop_index].vertex_index)
            self._egg_fp.write(
                '<Ref> { %s_Mesh }\n' % mesh_name +
                '%s  }\n' % padding +
                '%s}\n' % padding)

    def _write_groups(self, meshes, name):
        self._egg_fp.write('<Group> %s {\n' % name)
        indent_level = 2
        padding = indent_level*' '
        if len(self._human.data.bones) > 0:
            self._egg_fp.write('%s<Dart> { 1 }\n' % padding)
        self._egg_fp.write('%s<Group> CharacterRoot {\n' % padding)
        for mesh in meshes:
            self._write_mesh_object(mesh, indent_level+2)
        if len(self._human.data.bones) > 0:
            self._write_armature(meshes, name, indent_level+2)
        self._egg_fp.write(
            '%s}\n' % padding +
            '}\n')

    def _write_bone_translation(self, bone, indent_level=0):
        loc = bone.head_local
        padding = indent_level*' '
        # make relative if we can
        if bone.parent:
            loc = loc - bone.parent.head_local
        self._egg_fp.write(
            '%s<Translate> { ' % padding +
            ' %.5f %.5f %.5f }\n' % (loc.x, loc.y, loc.z))

    def produce_egg(self):
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.scene.objects.active = self._human
        meshes = [child
                  for child in self._human.children
                  if child.type == 'MESH']
        try:
            try:
                self._egg_fp = codecs.open(self._filepath, 'w',
                                           encoding='utf-8')
                print('Writing Egg file %s' % self._filepath)
            except:
                self._egg_fp = None
                print('Unable to open file for writing %s' % self._filepath)
            blendFilename = bpy.path.basename(bpy.context.blend_data.filepath)
            eggFilename = self._filename + '.egg'
            self._egg_fp.write('<CoordinateSystem> { Z-Up }\n\n' +
                               '<Comment> {\n' +
                               '"io_export_egg.py %s ' % blendFilename +
                               '%s"\n' % eggFilename +
                               '}\n\n')
            print('Exporting textures')
            self._write_textures(meshes)
            print('Exporting materials')
            self._write_materials(meshes)
            print('Exporting geometry & armature')
            name = self._filename
            self._write_groups(meshes, name)
        finally:
            if self._egg_fp:
                self._egg_fp.close()
            print('Done.')
        return {'FINISHED'}


##########################################
# ExportEggOperator class register/unregister
##########################################

class ExportEggOperator(bpy.types.Operator, ExportHelper):
    """Export selected object to Panda3D egg file"""
    bl_idname = "export.egg"
    bl_label = "Export to Panda3D egg"
    filename_ext = ".egg"
    filter_glob = StringProperty(default="*.egg", options={'HIDDEN'})
    vertex_membership_precision = EnumProperty(items=(
        ('4', "4 decimal places", ""),
        ('5', "5 decimal places", ""),
        ('6', "6 decimal places", ""),
        ),
        name="precision to",
        description="Vertex to bone membership precision",
        default='4')
    use_rel_paths = BoolProperty(
        name="use relative paths",
        description="Use relative path",
        default=True,
        )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label('Options:')
        box.prop(self, 'vertex_membership_precision')
        box.prop(self, 'use_rel_paths')

    @classmethod
    def poll(cls, context):
        selected = context.selected_objects
        return selected

    def execute(self, context):
        human = context.selected_objects[0]
        eggWorker = ExportEggWorker(human, self.vertex_membership_precision,
                                    self.filepath, self.use_rel_paths)
        return eggWorker.produce_egg()


def menu_func(self, context):
    self.layout.operator(ExportEggOperator.bl_idname, text="Panda3D (.egg)")


def register():
    bpy.utils.register_class(ExportEggOperator)
    bpy.types.INFO_MT_file_export.append(menu_func)


def unregister():
    bpy.utils.unregister_class(ExportEggOperator)
    bpy.types.INFO_MT_file_export.remove(menu_func)


if __name__ == "__main__":
    register()
