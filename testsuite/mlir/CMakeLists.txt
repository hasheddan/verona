configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(VERONA_OPT_TEST_DEPENDS "verona-mlir")

add_lit_testsuite(check-verona-mlir "Running the verona-mlir regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${VERONA_OPT_TEST_DEPENDS}
        )
      set_target_properties(check-verona-mlir PROPERTIES FOLDER "Tests" DEPENDS install)

add_lit_testsuites(VERONA_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${VERONA_OPT_TEST_DEPENDS})
