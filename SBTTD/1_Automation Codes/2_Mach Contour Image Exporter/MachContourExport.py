# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 16:46:46 2025

@author: hhsabbah
"""
import tecplot as tp
from pathlib import Path
import re 
from typing import Iterable, Optional
from typing import Iterable, Optional, Union
from os import fspath
from pathlib import PurePath, Path
### Functions #### 

def replace_line_by_prefix(
    file_path: Path,
    prefix: str,
    new_line: str,
    *,
    first_only: bool = True,
    case_sensitive: bool = True,
    encoding: str = "utf-8",
) -> int:
    """
    Replace line(s) that start with `prefix` (after optional leading spaces) with `new_line`.
    - Matches the whole line via regex; you only provide the identifying `prefix`.
    - Preserves the original leading indentation.
    - Returns the number of lines replaced (0 if none).
    """
    text = file_path.read_text(encoding=encoding)

    flags = re.MULTILINE
    if not case_sensitive:
        flags |= re.IGNORECASE

    # Capture leading spaces, then the literal prefix, then the rest of the line.
    pat = re.compile(rf"^([ \t]*){re.escape(prefix)}.*$", flags)

    # Keep original indentation (\1) and insert your replacement line (left-stripped).
    repl = lambda m: f"{m.group(1)}{new_line.lstrip()}"

    if first_only:
        new_text, n = pat.subn(repl, text, count=1)
    else:
        new_text, n = pat.subn(repl, text)

    if n:
        file_path.write_text(new_text, encoding=encoding)
    return n

def token_prefix_match(a: str, b: str) -> bool:
    ta = a.split('_')
    tb = b.split('_')
    # True if a is a token-prefix of b OR b is a token-prefix of a
    return tb[:len(ta)] == ta or ta[:len(tb)] == tb



Pathish = Union[str, Path]

def find_token_prefix_match(target: Pathish, items: Iterable[Pathish], *, case_sensitive: bool = True) -> Optional[Pathish]:
    """
    Return the first item in `items` whose basename matches `target` by token-prefix:
      - Split both names on '_' and compare tokens.
      - A match occurs if one name's tokens are a prefix of the other's.
        e.g., 'h_l_0.02' matches 'h_l_0.02_p0_7bar', and vice versa.
    Works with str or pathlib Paths. Returns the original matching item (same type) or None.
    """

    def base_name(x: Pathish) -> str:
        s = fspath(x)  # str path for both str/Path
        name = PurePath(s).name  # last component; if no separators, returns s
        return name if case_sensitive else name.casefold()

    def token_prefix(a: str, b: str) -> bool:
        ta, tb = a.split("_"), b.split("_")
        return ta == tb[:len(ta)] or tb == ta[:len(tb)]

    tname = base_name(target)
    for item in items:
        if token_prefix(base_name(item), tname):
            return item
    return None

### End Functions Section ###

### Creating contours through multiple simulations. The attempt here is to automate this process ### 
  
# Root directory to import mcfd_tec.bin files # 
rootDir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Results\Mach Study 2") # this is the root directory to the parametric study solution files
subDirs1 = [p for p in rootDir.iterdir() if p.is_dir()]


# Finding destination directories to export the mach contours to the desired location #
destRootDir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Results Contour\Mach Contour") #This is where you want your contours to be 
destSubDir1 = [p for p in destRootDir.iterdir() if p.is_dir()]





# Directory of the Tecplot Macro file #
macroDir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\10_Tecplot Journal Scripts\1_Exporting Contours\1_Exporting Mach Contour\machContourExport1.mcr") # This is the directory of the macro file that I created


# Using PyTecplot to open tecplot application automatically # 
tp.session.connect() # Connecting to the open tecplot application 
tp.new_layout() # Creating a new tecplot layout. 



# Converting Dest Paths into Strings # 
# 1) As plain string paths
str_paths = [str(p) for p in destSubDir1]           # or: [os.fspath(p) for p in paths]
# or with forward slashes regardless of OS:
destSubDir1_str = [p.as_posix() for p in destSubDir1]



# Pre-Allocating Variable # 
fileName = "mcfd_tec.bin"
subDirs2 = [p for d in subDirs1 for p in d.iterdir() if p.is_dir()]  # flattened



# Collect all matching files
subDirs2 = [p / fileName for p in subDirs2 if (p / fileName).is_file()]



for idx, subDir2 in enumerate(subDirs2):
    tp.data.load_tecplot(subDir2.as_posix())
    outputFileName = subDir2.parent.name + ".png" 
    currPathMatch = find_token_prefix_match(subDir2.parent.name ,destSubDir1)
    
    
    currExportPath = currPathMatch.as_posix() + "/" + outputFileName
    try:
        # Changing the Macro so it works for our case # 
        replace_line_by_prefix(
            macroDir,
            "$!ExportSetup ExportFName",
            rf"$!ExportSetup ExportFName = '{currExportPath}'"
            )
        
        # Executing the Macro Directory #
        tp.macro.execute_file(macroDir)
        
        # Creating a new layout # 
        tp.new_layout() # Creating a new tecplot layout. 
    except:
        print(f"{currExportPath} doesn't have a mcfd_tec.bin file...\n")
    

    
print("Script run complete!\n")
    






 