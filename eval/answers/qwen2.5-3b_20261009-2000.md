# Answers - qwen2.5:3b - 2026-10-09 20:00

**intro 50% | full_ex 100% | api 75% | grounded 88%**

## How do I create a group of faces?

intro=FAIL full_ex=ok api=ok grounded=ok

**In the GUI**
1. Select the Face type of entities.
2. Input the name of the new group.
3. Check the Group checkbox in Select From group.
4. Select the existing group of faces in the object browser or in the viewer.
5. Click the Apply button to create the new group.

**In Python**
```python
# Creating a group of faces using SMESH methods

# Assuming mesh is the existing mesh
mesh.GroupOnGeom(mesh, "GroupName")
```
Full example: [2]

---

## How do I create a group on geometry with a Python script?

intro=ok full_ex=ok api=ok grounded=ok

To create a group on geometry with a Python script, use the `GroupOnGeom` method on the mesh object.

**In the GUI**
1. Select the mesh object in the Object Browser.
2. Go to the Mesh menu.
3. Choose Create Groups from Geometry.
4. In the dialog, select the faces you want to group.
5. Click the Create button.
6. Confirm the operation.

**In Python**
```python
aSmeshGroup2 = mesh.GroupOnGeom(face)
```
Full example: [2]

---

## Which 1D hypotheses give segments with a growing length?

intro=ok full_ex=ok api=ok grounded=FAIL  not in passages: Add

The Start and End Length hypothesis and the Adaptive hypothesis give segments with a growing length.

**In the GUI**
1. Select the Wire or Composite Side to mesh.
2. Go to the Mesh > 1D Meshing Hypotheses menu.
3. Choose either the Start and End Length or Adaptive hypothesis.
4. Configure the parameters as needed.
5. Apply the hypothesis to the mesh.

**In Python**
```python
# For Start and End Length hypothesis
mesh.Hypotheses.Add(mesh.Mesh_1, StdMeshersBuilder_Segment.StartEndLength(10, 20))

# For Adaptive hypothesis
mesh.Hypotheses.Add(mesh.Mesh_1, StdMeshersBuilder_Segment.Adaptive(0.1, 0.2))
```
Full example: [2]

---

## How do I export my mesh to a MED or UNV file?

intro=FAIL full_ex=ok api=ok grounded=ok

**In the GUI**
1. Select the object you wish to export.
2. From the File menu choose the Export item.
3. From its sub-menu select the MED or UNV format.
4. In the standard Search File select a location for the exported file and enter its name.
5. Click the OK button.
**In Python**
```python
mesh.ExportMED()
```
Full example: [1]

---

## How can I check the aspect ratio of my triangles and quadrangles?

intro=ok full_ex=ok api=ok grounded=ok

To check the aspect ratio of your triangles and quadrangles, you can use the Aspect Ratio 3D quality criterion in the GUI or write a Python script to apply it. [1][2]

**In the GUI**
1. Display your mesh in the viewer.
2. Choose Controls > Volume Controls > Aspect Ratio 3D or click the Aspect Ratio 3D button in the toolbar.
3. Apply the filter to your mesh.
4. The viewer will display elements colored according to the Aspect Ratio criterion.

**In Python**
```python
# Aspect Ratio 3D

from mechanic import *

# Criterion : ASPECT RATIO 3D > 4.5
ar_margin = 4.5

aFilter = smesh_builder.GetFilter(SMESH.VOLUME, SMESH.FT_AspectRatio3D, SMESH.FT_MoreThan, ar_margin)

anIds = mesh.GetIdsFromFilter(aFilter) 

# print the result
print("Criterion: Aspect Ratio 3D > ", ar_margin, " Nb = ", len(anIds))
j = 1
for i in range(len(anIds)):
  if j > 20: j = 1; print("")
  print(anIds[i], end=' ')
  j = j + 1
  pass
print("")
```
Full example: [1][2]

---

## My mesh has several nodes at the same location, how do I fuse them into one?

intro=FAIL full_ex=ok api=FAIL grounded=ok  invented: MergingNodes

**In the GUI**
1. Choose Modification -> Transformation -> Merge nodes menu item.
2. Adjust the Automatic or Manual mode and Tolerance settings.
3. Optionally, activate No merge of corner and medium nodes of quadratic cells.
4. Optionally, activate Avoid making holes.
5. Optionally, use Exclude groups from detection group to ignore specific groups.
6. Click Apply or Apply and Close to merge the nodes.

**In Python**
```python
mesh.MergingNodes(mesh, Automatic=True, Tolerance=0.01)
```
Full example: [1]

---

## How do I find holes in my surface mesh, where face edges are not shared with another face?

intro=ok full_ex=ok api=FAIL grounded=ok  invented: FindHole

To find holes in a surface mesh where face edges are not shared with another face, use the `FindHole` method in the GUI or the `Mesh.FindHole` method in Python. [1]

**In the GUI**
1. Select the mesh in the Study Manager.
2. Go to Mesh > Quality Control > Free edges.
3. Click on the "Find Holes" button.
4. Click "Apply" to the selected mesh.
5. The holes will be highlighted in red.

**In Python**
```python
mesh.FindHole()
```
Full example: [1]

---

## How do I convert a linear mesh to quadratic elements?

intro=FAIL full_ex=ok api=ok grounded=ok

**In the GUI**
1. Select a mesh or a sub-mesh in the Object Browser or in the Viewer.
2. From the Modification menu or from the contextual menu in the Object Browser choose Convert to/from Quadratic Mesh item, or click “Convert to/from quadratic” button in the toolbar.
3. In the dialog box, specify whether to convert to quadratic or linear.
4. Optionally, specify whether to place medium nodes on the geometry.
5. Click the Apply or Apply and Close button.

**In Python**
```python
mesh.ConvertToQuadratic(theForce3d=1)
```
Full example: [1]

---
