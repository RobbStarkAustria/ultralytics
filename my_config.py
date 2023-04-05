training_size = 1024
augment_images = False
factor_of_augmentation = 3

# classes for normal training
# classes_to_train = ["bar", "clef", "custos", "s-b"]
# classes_to_train = ["flat", "dot", "sharp", "nat"]
# classes_to_train = ["fu-d", "fu-u"]
# classes_to_train = ["li-lolu", "br-min", "sb-min", "n3"]
# classes_to_train = ["ma-d", "ma-u", "lo-u", "lo-d", "lo-l-d"]
# classes_to_train = ["mens", "bre"]
# classes_to_train = ["mi-d", "mi-u"]
# classes_to_train = ["r-sm", "r-sb", "r-mi", "r-br", "r-lo", "r-fu", "r-ma"]
# classes_to_train = ["sebre", "sm-d", "sm-u"]
# classes_to_train = ["sf-d", "sf-u"]
# classes_to_train = ["staff"]
# classes_to_train = ["bar", "clef", "custos", "s-b",
#                     "dot", "sharp", "flat", "nat",
#                     "fu-u", "fu-d",
#                     "ma-d", "ma-u", "lo-d", "lo-u",
#                     "mens", "bre",
#                     "mi-u", "mi-d",
#                     "r-sm", "r-sb", "r-mi", "r-br", "r-lo", "r-fu", "r-ma",
#                     "sf-d", "sf-u",
#                     "sm-u", "sm-d", "sebre",
#                     "li-lolu", "br-min", "sb-min", "n3"]

note_group = ""
# notes to train
note_group ="diamant_notes"
classes_to_train = ["fu-d", "fu-u",
                    "sb-min",
                    "mi-d", "mi-u",
                    "sebre", "sm-d", "sm-u",
                    "sf-d", "sf-u"
                    ]

# note_group = "rectangular_notes"
# classes_to_train = ["li-lolu", "br-min",
#                     "ma-d", "ma-u", "lo-u", "lo-d", "lo-l-d",
#                     "bre"
#                     ]

# note_group = "rests"
# classes_to_train = ["r-sm", "r-sb", "r-mi", "r-br", "r-lo", "r-fu"]

# note_group = "extra_symbols"
# classes_to_train = ["bar", "clef", "custos", "s-b",
#                     "flat", "dot", "sharp", "nat",
#                     "n3", "mens"
#                     ]

# note_group = "all_symbols"
# classes_to_train = [
#     "fu-d", "fu-u", "sb-min",
#     "mi-d", "mi-u", "sebre",
#     "sm-d", "sm-u", "sf-d", "sf-u",
#     "li-lolu", "br-min", "ma-d",
#     "ma-u", "lo-u", "lo-d",
#     "lo-l-d", "bre",
#     "r-sm", "r-sb", "r-mi",
#     "r-sm", "r-sb", "r-mi",
#     "bar", "clef", "custos",
#     "s-b", "flat", "dot",
#     "sharp", "nat", "n3", "mens"
# ]

# note_group = "staff"
# classes_to_train = ["staff"]
