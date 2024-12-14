#This file belongs in the lut folder. This is the script that calls for the lookup tables to be generated.


MULTIPLIER=(
    "FMBM16_MULTIPLIER"
    "FMBM14_MULTIPLIER"
    "FMBM12_MULTIPLIER"
    "FMBM10_MULTIPLIER"
    "MITCHEL16_MULTIPLIER"
    "MITCHEL14_MULTIPLIER"
    "MITCHEL12_MULTIPLIER"
    "MITCHEL10_MULTIPLIER"
    "BFLOAT"
    "ZEROS"
    "ERRORINJECTOR"
    )
    for i in "${MULTIPLIER[@]}"; do
        g++ -D$i lut_gen.cc
        ./a.out
        done