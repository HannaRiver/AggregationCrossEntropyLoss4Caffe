# Install script for directory: /data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/src/caffe

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/cmake-build-debug/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/include/caffe")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/caffe/proto" TYPE FILE FILES "/data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/cmake-build-debug/include/caffe/proto/caffe.pb.h")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcaffe-d.so.1.0.0-rc5"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcaffe-d.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/cmake-build-debug/install/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial:/usr/local/cuda/lib64")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/cmake-build-debug/lib/libcaffe-d.so.1.0.0-rc5"
    "/data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/cmake-build-debug/lib/libcaffe-d.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcaffe-d.so.1.0.0-rc5"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcaffe-d.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/usr/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial:/usr/local/cuda/lib64:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
           NEW_RPATH "/data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/cmake-build-debug/install/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial:/usr/local/cuda/lib64")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/cmake-build-debug/lib/libproto-d.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe/proto" TYPE PROGRAM FILES
    "/data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/cmake-build-debug/include/caffe/proto/caffe_pb2.py"
    "/data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/cmake-build-debug/__init__.py"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/data_3/OCR_prject/LSTM训练方法/xxz/caffe-lstm-ocr/cmake-build-debug/src/caffe/test/cmake_install.cmake")

endif()

