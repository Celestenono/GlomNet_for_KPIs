def cross_validation_split(fold_num=0):
    fold_0 = ["12-116_wsi.tiff", "12-117_wsi.tiff", "11-356_wsi.tiff", "normal_F1_wsi.tiff", "normal_F2_wsi.tiff",
            "08-368_01_wsi.tiff", "08-368_02_wsi.tiff", "08-368_03_wsi.tiff", "08-373_01_wsi.tiff", "08-373_02_wsi.tiff"]
    fold_1 = ["12-169_wsi.tiff", "12-170_wsi.tiff", "11-357_wsi.tiff", "11-358_wsi.tiff", "normal_F3_wsi.tiff",
              "08-373_03_wsi.tiff", "08-471_01_wsi.tiff", "08-471_02_wsi.tiff", "08-471_03_wsi.tiff", "08-472_01_wsi.tiff"]
    fold_2 = ["12-171_wsi.tiff", "11-367_wsi.tiff", "11-370_wsi.tiff", "normal_F4_wsi.tiff", "normal_F1576_wsi.tiff",
              "08-472_02_wsi.tiff", "08-472_03_wsi.tiff", "08-474_01_wsi.tiff", "08-474_02_wsi.tiff","08-474_03_wsi.tiff"]
    test = ["12-173_wsi.tiff", "12-174_wsi.tiff", "11-359_wsi.tiff", "11-361_wsi.tiff", "18-575_wsi.tiff",
            "18-577_wsi.tiff", "normal_M1_wsi.tiff", "normal_M2_wsi.tiff"]

    # fold_0 = ["08-373_01_wsi.tiff", "08-373_02_wsi.tiff"]
    # fold_1 = ["08-471_03_wsi.tiff", "08-472_01_wsi.tiff"]
    # fold_2 = ["08-474_02_wsi.tiff", "08-474_03_wsi.tiff"]
    # test = ["12-173_wsi.tiff", "12-174_wsi.tiff", "11-359_wsi.tiff", "11-361_wsi.tiff", "18-575_wsi.tiff",
    #         "18-577_wsi.tiff", "normal_M1_wsi.tiff", "normal_M2_wsi.tiff"]

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