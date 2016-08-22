cmake_minimum_required(VERSION 2.8.3)

project(transforms3d NONE)

enable_testing()

find_package(PythonInterp REQUIRED)
find_program(NOSE nosetests)

if (NOT NOSE)
	message(FATAL_ERROR "could not find nosetests program")
endif()

set(setup_py ${CMAKE_CURRENT_SOURCE_DIR}/setup.py)

add_custom_target(python_build ALL ${PYTHON_EXECUTABLE}
	${setup_py} build
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	)

install(CODE "execute_process(
	COMMAND ${PYTHON_EXECUTABLE} ${setup_py} install
	--prefix=${CMAKE_INSTALL_PREFIX}
	--install-layout=deb
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	)")

install(FILES package.xml DESTINATION share/transforms3d)

add_test(nose ${NOSE} ${CMAKE_CURRENT_SOURCE_DIR}/transforms3d)
