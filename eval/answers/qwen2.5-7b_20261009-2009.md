# Answers - qwen2.5:7b - 2026-10-09 20:09

**intro 100% | full_ex 100% | api 100% | grounded 88%**

## How do I create a group of faces?

intro=ok full_ex=ok api=ok grounded=ok

To create a group of faces, you can use the Mesh menu's Create Group item in the GUI or the `FaceGroupsSeparatedByEdges` method in Python. [3][4]

**In the GUI**
1. Select the mesh in the Object Browser or 3D viewer.
2. In the Mesh menu, click Create Group.
3. Select Faces as the Elements Type.
4. Input a name for the group.
5. Click Apply.

**In Python**
```python
groups = mesh.FaceGroupsSeparatedByEdges(89)
```
Full example: [2]

---

## How do I create a group on geometry with a Python script?

intro=ok full_ex=ok api=ok grounded=ok

To create a group on geometry with a Python script, use the `GroupOnGeom` method on the mesh object [1].

**In the GUI**
1. Select the mesh in the Object browser.
2. Go to the Mesh menu and select Create Groups from Geometry.

**In Python**
```python
aSmeshGroup = mesh.GroupOnGeom(face, "SMESHGroup1")
```
Full example: [1]

---

## Which 1D hypotheses give segments with a growing length?

intro=ok full_ex=ok api=ok grounded=ok

The Start and End Length hypothesis gives segments with a growing length [4].

**In the GUI**
1. Go to Structured documentation > Defining hypotheses > 1D Meshing Hypotheses.
2. Select Start and End Length hypothesis.

**In Python**
```python
mesh.StartEndLength()
```
Full example: [4]

---

## How do I export my mesh to a MED or UNV file?

intro=ok full_ex=ok api=ok grounded=ok

To export your mesh to a MED or UNV file, you can use the File menu in the GUI or the appropriate Python method. [1]

**In the GUI**
1. Select the mesh or group you wish to export.
2. From the File menu, choose Export, then select MED or UNV from the sub-menu.

**In Python**
```python
mesh.ExportMED("output.med")
```
or

```python
mesh.ExportUNV("output.unv")
```

Full example: [1]

---

## How can I check the aspect ratio of my triangles and quadrangles?

intro=ok full_ex=ok api=ok grounded=ok

To check the aspect ratio of your triangles and quadrangles, you can use the Aspect Ratio and Aspect Ratio 3D quality criteria in the GUI or apply the corresponding Python methods.

**In the GUI**
1. Display your mesh in the viewer.
2. For triangles, choose Controls > Face Controls > Aspect Ratio or click the Aspect Ratio button in the toolbar.
3. For quadrangles, choose Controls > Volume Controls > Aspect Ratio 3D or click the "Aspect Ratio 3D" button in the toolbar.

**In Python**
```python
# For triangles
aFilter = smesh.GetFilter(SMESH.FACE, SMESH.FT_AspectRatio, SMESH.FT_MoreThan, 4.5)
tri_ids = mesh.GetIdsFromFilter(aFilter)

# For quadrangles
aFilter = smesh.GetFilter(SMESH.VOLUME, SMESH.FT_AspectRatio3D, SMESH.FT_MoreThan, 4.5)
quad_ids = mesh.GetIdsFromFilter(aFilter)
```
Full example: [1][2]

---

## My mesh has several nodes at the same location, how do I fuse them into one?

intro=ok full_ex=ok api=ok grounded=FAIL  not in passages: MergeNodes

To fuse nodes at the same location into one, use the "Merge nodes" functionality in the GUI or the `MergeNodes` method in Python. [1]

**In the GUI**
1. Choose Modification -> Transformation -> Merge nodes.
2. Adjust the Tolerance and other settings as needed.
3. Click Detect to generate the list of coincident nodes.
4. Review and adjust the groups of nodes to merge.
5. Click Apply to merge the nodes.

**In Python**
```python
mesh.MergeNodes(0.001)  # 0.001 is the tolerance value
```
Full example: [1]

---

## How do I find holes in my surface mesh, where face edges are not shared with another face?

intro=ok full_ex=ok api=ok grounded=ok

To find holes in your surface mesh, where face edges are not shared with another face, you can use the Free edges filter in the GUI or the `Mesh.FindFreeBorders()` method in Python. [1]

**In the GUI**
1. Go to Free edges > Free edges

**In Python**
```python
filter = mesh.GetFilter(SMESH.FACE, SMESH.FT_FreeEdges)
ids = mesh.GetIdsFromFilter(filter)
```
Full example: [4]

---

## How do I convert a linear mesh to quadratic elements?

intro=ok full_ex=ok api=ok grounded=ok

To convert a linear mesh to quadratic elements, use the Convert to/from Quadratic Mesh functionality in the GUI or the `ConvertToQuadratic` method in Python. [1]

**In the GUI**
1. Select the mesh in the Object Browser or in the Viewer.
2. From the Modification menu or from the contextual menu in the Object Browser, choose Convert to/from Quadratic Mesh, or click the "Convert to/from quadratic" button in the toolbar.
3. In the dialog box, specify that it is necessary to convert a linear mesh to quadratic.
4. Click the Apply or Apply and Close button.

**In Python**
```python
mesh.ConvertToQuadratic(theForce3d=1)
```

Full example: [2]

---
