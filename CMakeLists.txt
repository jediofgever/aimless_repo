cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
project( MSc_thesis )
find_package( OpenCV REQUIRED )


find_package(PCL 1.8 REQUIRED)



add_executable( Project3DBox2Img src/Project3DBox2Img.cpp )
target_link_libraries( Project3DBox2Img ${OpenCV_LIBS} )


add_executable( Project2DLabelToRGBPcl src/Project2DLabelToRGBPcl.cpp)
target_link_libraries( Project2DLabelToRGBPcl ${OpenCV_LIBS} )

add_executable (ProjectLidarPoint2Img src/ProjectLidarPoint2Img.cpp)
target_link_libraries (ProjectLidarPoint2Img ${PCL_LIBRARIES} ${OpenCV_LIBS} )

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

ADD_EXECUTABLE(bin2pcd src/bin2pcd.cpp)
TARGET_LINK_LIBRARIES(bin2pcd ${Boost_LIBRARIES} ${PCL_LIBRARIES})


 
add_executable (EuclidianCluster src/EuclidianCluster.cpp)
target_link_libraries (EuclidianCluster ${PCL_LIBRARIES}   )
 
if( PCL_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )
  
  # For Use Not PreCompiled Features 
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )

  # [Linker]>[Input]>[Additional Dependencies]
  target_link_libraries( bin2pcd ${PCL_LIBRARIES} )
endif()

add_executable (vis_pcd src/vis_pcd.cpp)
target_link_libraries (vis_pcd ${PCL_LIBRARIES})


