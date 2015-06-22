# Just an include to make main file cleaner
set(REPORT_DIR ${CMAKE_SOURCE_DIR}/../report)
  configure_file(
    ${CMAKE_SOURCE_DIR}/../report/ebof.tex ${CMAKE_CURRENT_BINARY_DIR}/ebof.tex @ONLY)
  configure_file(
    ${CMAKE_SOURCE_DIR}/../report/algorithm.tex ${CMAKE_CURRENT_BINARY_DIR}/algorithm.tex @ONLY)
  configure_file(
    ${CMAKE_SOURCE_DIR}/../report/code_design.tex ${CMAKE_CURRENT_BINARY_DIR}/code_design.tex @ONLY)
  configure_file(
    ${CMAKE_SOURCE_DIR}/../report/discussion.tex ${CMAKE_CURRENT_BINARY_DIR}/discussion.tex @ONLY)
  configure_file(
    ${CMAKE_SOURCE_DIR}/../report/experimental_results.tex ${CMAKE_CURRENT_BINARY_DIR}/experimental_results.tex @ONLY)
  configure_file(
    ${CMAKE_SOURCE_DIR}/../report/implementation.tex ${CMAKE_CURRENT_BINARY_DIR}/implementation.tex @ONLY)
  add_custom_target(step1
  COMMAND   ${LATEX_COMPILER} ${CMAKE_CURRENT_BINARY_DIR}/ebof.tex
  DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/ebof.tex
  COMMENT   "STEP1"
  )

  configure_file(
    ${CMAKE_SOURCE_DIR}/../report/ebof.bib ${CMAKE_CURRENT_BINARY_DIR}/ebof.bib @ONLY)
  add_custom_target(step2
  COMMAND   ${BIBTEX_COMPILER} ebof
  DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/ebof.aux
        ${CMAKE_CURRENT_BINARY_DIR}/ebof.bib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT   "STEP2: BIBTEX"
  )
  add_dependencies(step2 step1)

  add_custom_target(step3
  COMMAND   ${LATEX_COMPILER} ${CMAKE_CURRENT_BINARY_DIR}/ebof.tex
  DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/ebof.tex
  COMMENT   "STEP3"
  )
  add_dependencies(step3 step2)

  add_custom_target(step4
  COMMAND   ${LATEX_COMPILER} ${CMAKE_CURRENT_BINARY_DIR}/ebof.tex
  DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/ebof.tex
  COMMENT   "STEP4"
  )
  add_dependencies(step4 step3)

  IF(DVIPS_CONVERTER)
    add_custom_target(reportps
      COMMAND   ${DVIPS_CONVERTER} ${CMAKE_CURRENT_BINARY_DIR}/ebof.dvi
                -o ${CMAKE_CURRENT_BINARY_DIR}/ebof.ps
      DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/ebof.dvi
      COMMENT   "STEP5"
    )
    add_dependencies(reportps step4)

    IF(PS2PDF_CONVERTER)
      add_custom_target(report
      COMMAND   ${PS2PDF_CONVERTER} -sPAPERSIZE=a4 ${CMAKE_CURRENT_BINARY_DIR}/ebof.ps
      DEPENDS   ${CMAKE_CURRENT_BINARY_DIR}/ebof.ps
      COMMENT   "STEP6: Final PDF Report"
      )
      add_dependencies(report reportps)
    ENDIF(PS2PDF_CONVERTER)
  ENDIF(DVIPS_CONVERTER)