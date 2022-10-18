find_package(CUDA)


if(CUDA_FOUND)
  message("CUDA available!")
  message("CUDA Libs: ${CUDA_LIBRARIES}")
  message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
 
  list(APPEND CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_npps_LIBRARY} ${CUDA_nppig_LIBRARY})
  message("All togheter now (libs): ${CUDA_LIBRARIES}")


else()
  message("CUDA NOT Available")

endif()