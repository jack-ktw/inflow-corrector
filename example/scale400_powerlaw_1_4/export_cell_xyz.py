from paraview.simple import *
import glob, os

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'OpenFOAMReader'


foam_file = 'block.foam'
for fn in glob.glob("*.foam"):
    foam_file = fn
casefoam = OpenFOAMReader(FileName=foam_file)
casefoam.MeshRegions = ['inlet']
casefoam.CellArrays = ['U']
casefoam.CaseType = 'Decomposed Case'

cellCenters = CellCenters(Input=casefoam)

# Create a new 'SpreadSheet View'
spreadSheetView1 = CreateView('SpreadSheetView')
spreadSheetView1.ColumnToSort = ''
spreadSheetView1.BlockSize = 1024

cellCentersDisplay_1 = Show(cellCenters, spreadSheetView1)

# export view
ExportView('cell_xyz.csv', view=spreadSheetView1)
exit(0)