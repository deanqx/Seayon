if(BUILD_TESTING)
    get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    string(REPLACE " " "_" ProjectId ${ProjectId})
    project(${ProjectId})

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/${PROJECT_NAME}")
    add_executable(${PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/unit_test.cpp")

    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${INSTALL_DIR})
    
    target_link_libraries(${PROJECT_NAME} PRIVATE GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(${PROJECT_NAME})
endif()