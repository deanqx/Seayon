add_subdirectory("demo")
add_subdirectory("digit_recognition")

if(NOT CUDA_DISABLED)
    add_subdirectory("digit_recognition_gpu")
endif()