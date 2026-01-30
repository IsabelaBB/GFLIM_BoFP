add_executable(iftCreateGFLIMGraph iftCreateGFLIMGraph.c)
target_link_libraries(iftCreateGFLIMGraph ift)

add_executable(iftGFLIMConvertSeeds iftGFLIMConvertSeeds.c)
target_link_libraries(iftGFLIMConvertSeeds ift)

add_executable(iftGFLIMDecodeLayer iftGFLIMDecodeLayer.c)
target_link_libraries(iftGFLIMDecodeLayer ift)

add_executable(iftGFLIMExtract iftGFLIMExtract.c)
target_link_libraries(iftGFLIMExtract ift)

add_executable(iftGFLIMWriteActivations iftGFLIMWriteActivations.c)
target_link_libraries(iftGFLIMWriteActivations ift)

add_subdirectory(GFLIM)
add_subdirectory(GFLIM-BoFP)
add_subdirectory(GFLIM-BoFP-U)
add_subdirectory(M-GFLIM-BoFP)
add_subdirectory(GFLIM_BoFP_precompGraph)
