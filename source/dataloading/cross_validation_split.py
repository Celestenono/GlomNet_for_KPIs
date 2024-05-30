def cross_validation_split(fold_num=0):
    fold_0 = ["Biopsie_11.png", "Biopsie_10.png", "E_hNiere_S1_PAS.png", "Biopsie_12.png", "Biopsie_13.png",
            "A_hNiere_S1_PAS.png", "Biopsie_9.png", "Biopsie_17.png", "Biopsie_16.png", "Biopsie_8.png", "Biopsie_28.png",
              "01_00001_t=1_pas=01_s=1.png", "01_00001_t=1_pas=04_s=1.png", "01_00001_t=1_pas=06_s=1.png", "01_00001_t=1_pas=08_s=1.png",
             "01_00001_t=2_pas=01_s=1.png", "01_00001_t=2_pas=04_s=1.png", "01_00001_t=2_pas=06_s=1.png", "01_00001_t=2_pas=08_s=1.png",
             "01_00001_t=3_pas=01_s=1.png", "01_00001_t=3_pas=04_s=1.png", "01_00001_t=3_pas=06_s=1.png", "01_00001_t=3_pas=08_s=1.png",
              "01_00004_t=1_pas=01_s=1.png", "01_00004_t=1_pas=04_s=1.png", "01_00004_t=1_pas=06_s=1.png", "01_00004_t=1_pas=08_s=1.png",
              "01_00004_t=2_pas=01_s=1.png", "01_00004_t=2_pas=04_s=1.png", "01_00004_t=2_pas=06_s=1.png", "01_00004_t=2_pas=08_s=1.png",
              "01_00008_t=1_pas=01_s=1.png", "01_00008_t=1_pas=04_s=1.png", "01_00008_t=2_pas=01_s=1.png", "01_00008_t=2_pas=04_s=1.png",
              "01_00005_t=1_pas=1_s=1.png", "01_00005_t=1_pas=6_s=1.png", "01_00005_t=1_pas=8_s=1.png", "01_00005_t=2_pas=1_s=1.png",
              "01_00005_t=2_pas=6_s=1.png", "01_00005_t=2_pas=8_s=1.png", "01_00005_t=3_pas=1_s=1.png", "01_00005_t=3_pas=6_s=1.png",
              "01_00005_t=3_pas=8_s=1.png", "01_00007_t=1_pas=1_s=1.png", "01_00007_t=1_pas=4_s=1.png", "01_00007_t=1_pas=6_s=1.png",
              "01_00007_t=1_pas=8_s=1.png", "01_00007_t=2_pas=1_s=1.png", "01_00007_t=2_pas=4_s=1.png", "01_00007_t=2_pas=6_s=1.png",
              "01_00007_t=2_pas=8_s=1.png"] #"human_kidney_Bl_1_PAS.png",
    fold_1 = ["Biopsie_14.png", "Biopsie_15.png", "Biopsie_29.png", "C_hNiere_S1_PAS.png", "J_hNiere_S1_PAS.png",
              "Transplantniere_Bl_4_PAS.png", "K_hNiere_S1_PAS.png", "Biopsie_24.png", "Biopsie_6.png", "Biopsie_30.png", "Biopsie_18.png",
              "Biopsie_19.png", "01_00012_t=1_pas=01_s=1.png", "01_00012_t=1_pas=06_s=1.png", "01_00012_t=1_pas=08_s=1.png", "01_00012_t=1_pas=12_s=1.png",
              "01_00049_t=1_pas=01_s=1.png", "01_00049_t=1_pas=04_s=1.png", "01_00049_t=1_pas=06_s=1.png", "01_00049_t=1_pas=08_s=1.png",
              "01_00049_t=2_pas=01_s=1.png", "01_00049_t=2_pas=04_s=1.png", "01_00049_t=2_pas=06_s=1.png", "01_00049_t=2_pas=08_s=1.png",
              "01_00049_t=3_pas=01_s=1.png", "01_00049_t=3_pas=04_s=1.png", "01_00049_t=3_pas=06_s=1.png", "01_00049_t=3_pas=08_s=1.png",
              "01_00069_t=1_pas=01_s=1.png", "01_00069_t=1_pas=04_s=1.png", "01_00069_t=1_pas=06_s=1.png", "01_00069_t=1_pas=08_s=1.png",
              "01_00069_t=2_pas=01_s=1.png", "01_00069_t=2_pas=04_s=1.png", "01_00069_t=2_pas=06_s=1.png", "01_00069_t=2_pas=08_s=1.png",
              "01_00006_t=1_pas=1_s=1.png", "01_00006_t=1_pas=4_s=1.png", "01_00006_t=1_pas=6_s=1.png", "01_00006_t=1_pas=8_s=1.png",
              "01_00006_t=2_pas=1_s=1.png", "01_00006_t=2_pas=4_s=1.png", "01_00006_t=2_pas=4_s=2.png", "01_00006_t=2_pas=6_s=1.png",
              "01_00006_t=2_pas=8_s=1.png", "01_00006_t=3_pas=1_s=1.png", "01_00006_t=3_pas=4_s=1.png", "01_00006_t=3_pas=6_s=1.png",
              "01_00006_t=3_pas=8_s=1.png", "01_00006_t=4_pas=1_s=1.png", "01_00006_t=4_pas=4_s=1.png", "01_00006_t=4_pas=6_s=1.png",
              "01_00006_t=4_pas=8_s=1.png"]
    fold_2 = ["Biopsie_7.png", "Biopsie_31.png", "Biopsie_25.png", "H_hNiere_S1_PAS.png", "Biopsie_5.png", "Biopsie_27.png",
              "A_hNiere_S4_PAS.png", "Biopsie_26.png", "Biopsie_4.png","Biopsie_22.png", "Biopsie_23.png", "I_hNiere_S1_PAS.png",
              "01_00016_t=1_pas=01_s=1.png", "01_00016_t=1_pas=04_s=1.png", "01_00016_t=1_pas=06_s=1.png", "01_00016_t=1_pas=08_s=1.png",
              "01_00016_t=2_pas=01_s=1.png", "01_00016_t=2_pas=04_s=1.png", "01_00016_t=2_pas=06_s=1.png", "01_00016_t=2_pas=08_s=1.png",
              "01_00016_t=3_pas=01_s=1.png", "01_00016_t=3_pas=04_s=1.png", "01_00016_t=3_pas=06_s=1.png", "01_00044_t=1_pas=01_s=1.png",
              "01_00044_t=1_pas=04_s=1.png", "01_00044_t=1_pas=06_s=1.png", "01_00044_t=1_pas=08_s=1.png", "01_00044_t=2_pas=01_s=1.png",
              "01_00044_t=2_pas=04_s=1.png", "01_00044_t=2_pas=06_s=1.png", "01_00044_t=2_pas=08_s=1.png", "01_00044_t=3_pas=06_s=1.png",
              "01_00044_t=3_pas=08_s=1.png", "01_00050_t=1_pas=06_s=1.png", "01_00050_t=1_pas=08_s=1.png", "01_00050_t=1_pas=19_s=1.png",
              "01_00050_t=2_pas=19_s=2.png", "01_00010_t=1_pas=01_s=1.png", "01_00010_t=1_pas=01_s=2.png", "01_00010_t=1_pas=06_s=1.png",
              "01_00010_t=1_pas=06_s=2.png", "01_00010_t=1_pas=08_s=1.png", "01_00010_t=1_pas=08_s=2.png", "01_00010_t=1_pas=12_s=1.png",
              "01_00010_t=1_pas=12_s=2.png", "01_00010_t=2_pas=01_s=1.png", "01_00010_t=2_pas=01_s=2.png", "01_00010_t=2_pas=06_s=1.png",
              "01_00010_t=2_pas=06_s=2.png", "01_00010_t=2_pas=08_s=1.png", "01_00010_t=2_pas=08_s=2.png", "01_00010_t=2_pas=12_s=1.png",
              "01_00010_t=2_pas=12_s=2.png"]
    test = ["Biopsie_1.png", "Biopsie_21.png", "Biopsie_3.png", "Biopsie_2.png", "Biopsie_20.png",
            "01_00013_t=1_pas=01_s=1.png", "01_00013_t=1_pas=01_s=2.png", "01_00013_t=1_pas=06_s=1.png", "01_00013_t=1_pas=06_s=2.png",
            "01_00013_t=1_pas=08_s=1.png", "01_00013_t=1_pas=08_s=2.png", "01_00013_t=1_pas=12_s=1.png", "01_00013_t=1_pas=12_s=2.png",
            "01_00013_t=2_pas=01_s=1.png", "01_00013_t=2_pas=01_s=2.png", "01_00013_t=2_pas=06_s=1.png", "01_00013_t=2_pas=06_s=2.png",
            "01_00013_t=2_pas=08_s=1.png", "01_00013_t=2_pas=08_s=2.png", "01_00013_t=2_pas=12_s=1.png", "01_00013_t=2_pas=12_s=2.png",
            "01_00013_t=3_pas=01_s=1.png", "01_00013_t=3_pas=01_s=2.png", "01_00013_t=3_pas=06_s=1.png", "01_00013_t=3_pas=06_s=2.png",
            "01_00013_t=3_pas=08_s=1.png", "01_00013_t=3_pas=08_s=2.png", "01_00013_t=3_pas=12_s=1.png", "01_00013_t=3_pas=12_s=2.png",
            "01_00034_t=1_pas=01_s=1.png", "01_00034_t=1_pas=01_s=2.png", "01_00034_t=1_pas=06_s=1.png", "01_00034_t=1_pas=06_s=2.png",
            "01_00034_t=1_pas=08_s=1.png", "01_00034_t=1_pas=08_s=2.png", "01_00034_t=1_pas=19_s=1.png", "01_00039_t=1_pas=01_s=1.png",
            "01_00039_t=1_pas=04_s=1.png", "01_00039_t=1_pas=06_s=1.png", "01_00039_t=1_pas=08_s=1.png", "01_00039_t=2_pas=01_s=1.png",
            "01_00039_t=2_pas=04_s=1.png", "01_00039_t=2_pas=06_s=1.png", "01_00039_t=2_pas=08_s=1.png", "01_00039_t=3_pas=01_s=1.png",
            "01_00039_t=3_pas=04_s=1.png", "01_00039_t=3_pas=06_s=1.png", "01_00039_t=3_pas=08_s=1.png", "01_00059_t=1_pas=01_s=1.png",
            "01_00059_t=1_pas=04_s=1.png", "01_00059_t=1_pas=06_s=1.png", "01_00059_t=1_pas=08_s=1.png", "01_00059_t=2_pas=01_s=1.png",
            "01_00059_t=2_pas=04_s=1.png", "01_00059_t=2_pas=06_s=1.png", "01_00059_t=2_pas=08_s=1.png", "01_00059_t=3_pas=01_s=1.png",
            "01_00059_t=3_pas=04_s=1.png", "01_00059_t=3_pas=06_s=1.png", "01_00059_t=3_pas=08_s=1.png", "01_00059_t=4_pas=01_s=1.png",
            "01_00059_t=4_pas=04_s=1.png", "01_00059_t=4_pas=06_s=1.png", "01_00059_t=4_pas=08_s=1.png"]

    # fold_0 = ["Biopsie_11.png", "Biopsie_10.png", "human_kidney_Bl_1_PAS.png", "01_00008_t=2_pas=04_s=1.png"]
    # fold_1 = ["Biopsie_14.png", "Biopsie_15.png", "01_00069_t=2_pas=04_s=1.png", "01_00069_t=2_pas=06_s=1.png", "01_00069_t=2_pas=08_s=1.png"]
    # fold_2 = ["Biopsie_7.png", "Biopsie_31.png", "01_00050_t=1_pas=08_s=1.png", "01_00050_t=1_pas=19_s=1.png",
    #           "01_00050_t=2_pas=19_s=2.png"]
    # test = ["01_00013_t=2_pas=06_s=1.png", "01_00013_t=2_pas=06_s=2.png"]
    #
    # fold_0 = ["Biopsie_11.png"]
    # fold_1 = ["01_00013_t=2_pas=06_s=2.png"] #["01_00069_t=2_pas=06_s=1.png", "01_00013_t=2_pas=06_s=2.png"]
    # fold_2 = ["Biopsie_7.png"]
    # test = ["Biopsie_20.png"]
    #
    # test = ["01_00059_t=3_pas=01_s=1.png",
    #         "01_00059_t=3_pas=04_s=1.png", "01_00059_t=3_pas=06_s=1.png", "01_00059_t=3_pas=08_s=1.png",
    #         "01_00059_t=4_pas=01_s=1.png",
    #         "01_00059_t=4_pas=04_s=1.png", "01_00059_t=4_pas=06_s=1.png", "01_00059_t=4_pas=08_s=1.png"]

    if fold_num == 0:
        train = fold_1 + fold_2
        val = fold_0
    elif fold_num == 1:
        train = fold_0 + fold_2
        val = fold_1
    elif fold_num == 2:
        train = fold_0 + fold_1
        val = fold_2
    return {"train": train, "val": val, "test": test}