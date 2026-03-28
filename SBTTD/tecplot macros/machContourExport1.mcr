#!MC 1410
$!GlobalRGB RedChannelVar = 9
$!GlobalRGB GreenChannelVar = 3
$!GlobalRGB BlueChannelVar = 3
$!SetContourVar 
  Var = 4
  ContourGroup = 1
  LevelInitMode = ResetToNice
$!SetContourVar 
  Var = 5
  ContourGroup = 2
  LevelInitMode = ResetToNice
$!SetContourVar 
  Var = 9
  ContourGroup = 3
  LevelInitMode = ResetToNice
$!SetContourVar 
  Var = 10
  ContourGroup = 4
  LevelInitMode = ResetToNice
$!SetContourVar 
  Var = 11
  ContourGroup = 5
  LevelInitMode = ResetToNice
$!SetContourVar 
  Var = 12
  ContourGroup = 6
  LevelInitMode = ResetToNice
$!SetContourVar 
  Var = 13
  ContourGroup = 7
  LevelInitMode = ResetToNice
$!SetContourVar 
  Var = 14
  ContourGroup = 8
  LevelInitMode = ResetToNice
$!FieldLayers ShowContour = Yes
$!SetContourVar 
  Var = 10
  ContourGroup = 1
  LevelInitMode = ResetToNice
$!GlobalContour 1  ColorMapFilter{ColorMapDistribution = Continuous}

$!GlobalContour 1  ColorMapName = 'Diverging - Blue/Red'
$!FieldLayers ShowMesh = Yes
$!FieldLayers ShowMesh = No
$!ExportSetup ImageWidth = 1080
$!FrameLayout ShowBorder = No
$!Pick AddAtPosition
  X = 1.49135135135
  Y = 4.35810810811
  ConsiderStyle = Yes
$!TwoDAxis YDetail{Title{TitleMode = UseText}}
$!TwoDAxis YDetail{Title{Text = 'Y [m]'}}
$!Pick AddAtPosition
  X = 1.64702702703
  Y = 4.27162162162
  ConsiderStyle = Yes
$!Pick AddAtPosition
  X = 4.82108108108
  Y = 7.19486486486
  ConsiderStyle = Yes
$!Pick AddAtPosition
  X = 5.53027027027
  Y = 7.72243243243
  ConsiderStyle = Yes
$!TwoDAxis XDetail{Title{TitleMode = UseText}}
$!TwoDAxis XDetail{Title{Text = 'X [m]'}}
$!Pick AddAtPosition
  X = 5.10648648649
  Y = 4.53972972973
  ConsiderStyle = Yes
$!Pick AddAtPosition
  X = 2.19189189189
  Y = 3.61432432432
  ConsiderStyle = Yes
$!TwoDAxis YDetail{Ticks{ShowOnAxisLine = No}}
$!Pick AddAtPosition
  X = 4.75189189189
  Y = 7.35918918919
  ConsiderStyle = Yes
$!Pick AddAtPosition
  X = 8.79945945946
  Y = 7.48027027027
  ConsiderStyle = Yes
$!Pick AddAtPosition
  X = 8.73891891892
  Y = 7.35054054054
  ConsiderStyle = Yes
$!TwoDAxis XDetail{Ticks{ShowOnAxisLine = No}}
$!Pick AddAtPosition
  X = 5.39189189189
  Y = 5.04135135135
  ConsiderStyle = Yes
$!GlobalContour 1  Legend{IsVertical = No}
$!GlobalContour 1  Legend{Box{BoxType = None}}
$!ExportSetup ExportFName = 'C:/Users/hhsabbah/Documents/01_Bladeless_Proj/21_ANSYS Workflow Automation/8_Mach_Sweep_Study_2(Solution)/1_Processed Results/1_Visual Data/h_l_0.09/h_l_0.09_Mach_4.5.png'
$!Export 
  ExportRegion = AllFrames
