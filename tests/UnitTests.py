from labeler import Labeler, Selector
test_file = "../data/TCSSimpleWalk/basic_test.csv"

def test_labeler():
    labeler_ = Labeler.Labeler()
    modified_df = labeler_.add_direction(test_file)
    print('done')

test_labeler()



