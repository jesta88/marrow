# Runnable examples. These are NOT part of the zero-dependency runtime — they link the
# marrow library and demonstrate the public API. The quickstart doubles as a documentation
# snippet (quoted into site/tutorials/first-character.md) and as a CTest, so the code in
# the docs is always code that compiles and runs.

add_executable(mrw_quickstart quickstart.c)
target_link_libraries(mrw_quickstart PRIVATE marrow)
set_target_properties(mrw_quickstart PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

# Run it against the committed golden fixture so a broken load path fails CI everywhere
# ctest runs (which is every platform in the matrix).
if (MRW_BUILD_TESTS)
  add_test(NAME example_quickstart
           COMMAND mrw_quickstart ${CMAKE_SOURCE_DIR}/fixtures/rig.mrw)
endif()
