cmake_minimum_required (VERSION 3.1)
set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(PETSC_DIR "/home/nero/petsc" CACHE PATH "Path to SLEPc")
set(PETSC_ARCH "linux-gnu" CACHE STRING "PETSc architecture")
set(SLEPC_DIR "/home/nero/slepc" CACHE PATH "Path to SLEPc")
set(MPI_DIR "/opt/intel/oneapi/mpi/latest" CACHE PATH "Path to MPI")
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

set (CMAKE_C_COMPILER mpicc)
set (CMAKE_CXX_COMPILER mpicxx)

set(USE_PETSC True)
set(USE_SLEPC True)
set(USE_MPI True)

include (CheckSymbolExists)
include (CheckCXXSymbolExists)
include (CheckIncludeFiles)
include (CheckIncludeFileCXX)
include (CheckCXXSourceCompiles)
include (GenerateExportHeader)
include (ProcessorCount)
include (GenerateVersion)

project (PMTop CXX C)


string(TIMESTAMP YEAR "%Y")
set (oofem_COPYRIGHT "Copyright (C) 1994-${YEAR} Huang Kai")


if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Debug)
endif ()

set(CMAKE_VERBOSE_MAKEFILE ON)


function(add_git_submodule dir)
# add a Git submodule directory to CMake, assuming the
# Git submodule directory is a CMake project.
#
# Usage: in CMakeLists.txt
# 
# include(AddGitSubmodule.cmake)
# add_git_submodule(mysubmod_dir)
  find_package(Git REQUIRED)
  if(NOT EXISTS ${dir}/CMakeLists.txt)
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- ${dir}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      #COMMAND_ERROR_IS_FATAL ANY #cmake 3.19
      )
  endif()
  add_subdirectory(${dir})
endfunction(add_git_submodule)

if (USE_SLEPC)
    set (USE_PETSC ON)
endif ()

if (USE_PETSC)
    if (NOT "${PETSC_DIR}")
        # setting defaults for typical linux environments:
        #file (GLOB paths "/usr/lib/petscdir/*.*.*")
        #list (GET paths 0 PETSC_DIR)
        #message ("***************** PETSC_DIR: ${PETSC_DIR}")
    endif ()
    if (NOT "${PETSC_ARCH}")
        #file (GLOB children RELATIVE ${PETSC_DIR} "${PETSC_DIR}/*")
        #list (GET children 0 PETSC_ARCH)
        #message ("***************** PETSC_ARCH: ${PETSC_ARCH}")
    endif ()
    find_package (PETSc)
    # find_package(PETSc COMPONENTS CXX)
    include_directories (${PETSC_INCLUDES})
    list (APPEND EXT_LIBS ${PETSC_LIBRARIES})
    list (APPEND MODULE_LIST "PETSc")
    set (USE_MPI ON)
    add_definitions (-D__PETSC_MODULE)
endif ()

if (USE_SLEPC)
    if (SLEPC_DIR)
        find_library (SLEPC_LIBS slepc PATH "${SLEPC_DIR}/${PETSC_ARCH}/lib")
        include_directories ("${SLEPC_DIR}/include")
        include_directories ("${SLEPC_DIR}/${PETSC_ARCH}/include")
    else ()
        find_library (SLEPC_LIBS slepc)
    endif ()
    if (${SLEPC_LIBS} STREQUAL "SLEPC_LIBS-NOTFOUND")
        message (FATAL_ERROR "SLEPc library not found")
    endif ()
    list (APPEND EXT_LIBS ${SLEPC_LIBS})
    list (APPEND MODULE_LIST "slepc")
    set (USE_PETSC ON)
    add_definitions (-D__SLEPC_MODULE)
endif ()


if (USE_MPI)
    if (MPI_DIR)
        include_directories ("${MPI_DIR}/include")
        find_library (MPI_LIBRARY NAMES mpi PATHS ${MPI_DIR}/lib64 ${MPI_DIR}/lib)
        find_library (MPI_CXX_LIBRARY NAMES mpi_cxx PATHS ${MPI_DIR}/lib64 ${MPI_DIR}/lib)
        list (APPEND EXT_LIBS ${MPI_LIBRARY})
        if (MPI_CXX_LIBRARY)
            list (APPEND EXT_LIBS ${MPI_CXX_LIBRARY})
        endif ()
    else ()
        include (FindMPI)
        find_package (MPI REQUIRED)
        set (CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILES_FLAGS} ${MPI_COMPILE_FLAGS})
        set (CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
        add_definitions (-DMPICH_IGNORE_CXX_SEEK)
        include_directories (${MPI_INCLUDE_PATH})
        list (APPEND EXT_LIBS ${MPI_LIBRARIES})
    endif (MPI_DIR)
    list (APPEND MODULE_LIST "MPI")
    # Compile and link flags?
endif ()

add_executable(PMTop main.cpp)


target_link_libraries (PMTop ${EXT_LIBS})