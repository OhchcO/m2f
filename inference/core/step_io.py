# -*- coding: utf-8 -*-
"""STEP 文件 I/O 工具函数：读取 STEP、OCC 面→网格转换。"""

import numpy as np
import pyvista as pv

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopoDS import topods
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere


def _triangulation(face, location):
    return BRep_Tool.Triangulation(face, location)


def _face(shape):
    return topods.Face(shape)


try:
    import vtk
except ImportError:
    vtk = None


# ==================== OCC 面 → 网格转换 ====================


def face_to_pyvista_mesh(face):
    """将 OCC Face 转换为 pyvista PolyData 网格。"""
    location = TopLoc_Location()
    triangulation = _triangulation(face, location)
    if triangulation is None or triangulation.NbTriangles() == 0:
        return None

    transform = location.Transformation()
    points = []
    for i in range(1, triangulation.NbNodes() + 1):
        point = triangulation.Node(i)
        try:
            point = point.Transformed(transform)
        except Exception:
            pass
        points.append([point.X(), point.Y(), point.Z()])

    faces_pv = []
    is_reversed = face.Orientation() == TopAbs_REVERSED
    for i in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(i)
        n1 = tri.Value(1) - 1
        n2 = tri.Value(2) - 1
        n3 = tri.Value(3) - 1
        if is_reversed:
            n2, n3 = n3, n2
        faces_pv.append([3, n1, n2, n3])

    return pv.PolyData(np.array(points), np.array(faces_pv, dtype=np.int64))


def face_to_polydata(face):
    """将 OCC Face 转换为 VTK PolyData 网格（含法线一致性处理）。

    适用于需要 VTK 格式的场景（如交互式标注工具）。
    """
    if vtk is None:
        raise ImportError("vtk is required for face_to_polydata")

    location = TopLoc_Location()
    triangulation = _triangulation(face, location)
    if triangulation is None or triangulation.NbTriangles() == 0:
        return None

    transform = location.Transformation()
    points = vtk.vtkPoints()
    for i in range(1, triangulation.NbNodes() + 1):
        point = triangulation.Node(i)
        try:
            point = point.Transformed(transform)
        except Exception:
            pass
        points.InsertNextPoint(point.X(), point.Y(), point.Z())

    cells = vtk.vtkCellArray()
    is_reversed = face.Orientation() == TopAbs_REVERSED
    for i in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(i)
        n1 = tri.Value(1) - 1
        n2 = tri.Value(2) - 1
        n3 = tri.Value(3) - 1
        if is_reversed:
            n2, n3 = n3, n2
        vtk_tri = vtk.vtkTriangle()
        vtk_tri.GetPointIds().SetId(0, n1)
        vtk_tri.GetPointIds().SetId(1, n2)
        vtk_tri.GetPointIds().SetId(2, n3)
        cells.InsertNextCell(vtk_tri)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    output = vtk.vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    return output


def get_face_type(face):
    """返回 OCC Face 的几何类型名称。"""
    surf = BRepAdaptor_Surface(face)
    stype = surf.GetType()
    if stype == GeomAbs_Plane:
        return "Plane"
    if stype == GeomAbs_Cylinder:
        return "Cylinder"
    if stype == GeomAbs_Cone:
        return "Cone"
    if stype == GeomAbs_Sphere:
        return "Sphere"
    return "Other"


def load_step_faces(step_path):
    """读取 STEP 文件，提取所有面的网格和信息。

    Returns:
        dict: {"faces": [...], "bounds": [...], "shape": shape}，读取失败返回 None。
    """
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print(f"  Error reading STEP: {step_path}")
        return None

    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.1)

    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 1
    while exp.More():
        face = _face(exp.Current())
        face_type = get_face_type(face)
        mesh = face_to_pyvista_mesh(face)
        if mesh is not None:
            faces.append({
                "face_id": face_id,
                "face_type": face_type,
                "mesh": mesh,
            })
        face_id += 1
        exp.Next()

    bounds = [1e9, -1e9, 1e9, -1e9, 1e9, -1e9]
    for item in faces:
        fb = item["mesh"].bounds
        bounds[0] = min(bounds[0], fb[0])
        bounds[1] = max(bounds[1], fb[1])
        bounds[2] = min(bounds[2], fb[2])
        bounds[3] = max(bounds[3], fb[3])
        bounds[4] = min(bounds[4], fb[4])
        bounds[5] = max(bounds[5], fb[5])

    return {"faces": faces, "bounds": bounds, "shape": shape}
