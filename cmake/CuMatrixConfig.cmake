include(${CMAKE_CURRENT_LIST_DIR}/FindCuda.cmake)

if (CUDA_FOUND)
  message("Find Cuda SUCCESS!\n")
  set(CUDA_FOUND ON)
else()
  message("Cuda NOT FOUND\n")

endif (CUDA_FOUND)

# complain if no backend is installed
if(NOT CUDA_FOUND)
  message(FATAL_ERROR
  "CUDA must be installed
  CUDA_FOUND ${CUDA_FOUND}\n")
endif()


SET (CU_MATRIX_INCLUDE_DIR 
	${CUDA_INCLUDE_DIRS}
	${CMAKE_CURRENT_LIST_DIR}/../	
)

SET (CU_MATRIX_LIBS
	${CUDA_LIBRARIES}	
)



SET (CU_MATRIX_SOURCE_CPP
	#
)

