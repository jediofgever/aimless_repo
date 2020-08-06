cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
project( MSc_thesis )
find_package( OpenCV REQUIRED )

find_package(PCL 1.8 REQUIRED)

add_executable( Project2DLabelToRGBPcl src/Project2DLabelToRGBPcl.cpp)
target_link_libraries( Project2DLabelToRGBPcl ${OpenCV_LIBS} )

if( PCL_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )
  
  # For Use Not PreCompiled Features 
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )

 
endif()
