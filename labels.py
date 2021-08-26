class MaskLabels:
    mask = 0
    incorrect = 1
    normal = 2

    
class GenderLabels:
    male = 0
    female = 1

    
class AgeGroup:
    map_label = lambda x:0 if int(x) < 30 else 1 if int(x) < 60 else 2