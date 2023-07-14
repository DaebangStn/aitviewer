import os
import shutil

import numpy as np
from PIL import Image
from pxr import Sdf, UsdShade


def _get_texture_paths(path, name, directory):
    usd_path = os.path.join("textures", name)
    fs_path = os.path.join(directory, usd_path)
    os.makedirs(fs_path, exist_ok=True)
    return os.path.join(fs_path, os.path.basename(path)), os.path.join(usd_path, os.path.basename(path))


def copy_texture(path, name, directory):
    fs_path, usd_path = _get_texture_paths(path, name, directory)
    shutil.copy(path, fs_path)
    return usd_path


def save_image_as_texture(img, img_name, name, directory):
    fs_path, usd_path = _get_texture_paths(img_name, name, directory)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(fs_path)
    return usd_path


def add_texture(stage, mesh, usd_path, texture_path):
    # Material.
    mat_path = usd_path + "/material"
    material = UsdShade.Material.Define(stage, mat_path)
    input = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
    input.Set("st")

    # Shader.
    shader = UsdShade.Shader.Define(stage, mat_path + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")

    # Connect the material to the shader.
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Create a texture coordinates reader and connect it to the material input.
    reader = UsdShade.Shader.Define(stage, mat_path + "/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(input)

    # Create a texture.
    diffuse = UsdShade.Shader.Define(stage, mat_path + "/diffuseTexture")
    diffuse.CreateIdAttr("UsdUVTexture")
    diffuse.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
    diffuse.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    diffuse.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuse.ConnectableAPI(), "rgb")

    # Bind the Material to the mesh.
    mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
