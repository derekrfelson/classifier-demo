cmake_minimum_required(VERSION 3.2.2)
project (classifier)
project (classifier-tests)

find_package(Threads)

include_directories(src)
file(GLOB TESTS "tests/*.cpp")
link_directories(/usr/local/lib)
set(SOURCES src/Classifier.cpp src/Dataset.cpp src/Partition.cpp)
set(MAINSOURCE src/classifier-demo.cpp)

add_executable(classifier ${SOURCES} ${MAINSOURCE})
add_executable(classifier-tests ${SOURCES} ${TESTS})
target_link_libraries(classifier-tests /usr/local/lib/gtest_main.a ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET classifier PROPERTY CXX_STANDARD 14)
set_property(TARGET classifier-tests PROPERTY CXX_STANDARD 14)
